#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsemax import Sparsemax
from sinkhorn import Sinkhorn
from util import knn
import math
from model import PaiConv


class PaiNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PaiNet, self).__init__()
        self.args = args
        self.k = args.k
        num_kernel = 9 # xyz*3 + 1
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        map_size = 32
        num_bases = 16
        self.B = nn.Parameter(torch.randn(7*self.k, map_size) , requires_grad=False) 
        self.mlp = nn.Linear(map_size*2, num_bases, bias=False)
        self.permatrix = nn.Parameter(torch.randn(num_bases, self.k, num_kernel), requires_grad=True)
        self.permatrix.data = torch.cat([torch.eye(num_kernel), torch.zeros(self.k-num_kernel, num_kernel)], 
                                        dim=0).unsqueeze(0).expand_as(self.permatrix) 
        self.softmax = Sparsemax(dim=-1) 

        self.conv1 = PaiConv(3, 64, self.k, num_kernel)
        self.conv2 = PaiConv(64, 64, self.k, num_kernel)
        self.conv3 = PaiConv(64, 128, self.k, num_kernel)
        self.conv4 = PaiConv(128, 256, self.k, num_kernel)

        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5)
        
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def permatrix_lsa(self, x):
        bsize, num_feat, num_pts = x.size()

        neigh_indexs = knn(x, self.k)
        idx_base = torch.arange(0, bsize, device=x.device).view(-1, 1, 1)*num_pts
        neigh_indexs = (neigh_indexs + idx_base).view(-1) # bsize*num_pts*spiral_size

        x_feats = x.permute(0, 2, 1).contiguous().view(bsize*num_pts, -1)
        x_feats = x_feats[neigh_indexs,:].view(bsize*num_pts, self.k, -1)
        x_repeat = x_feats[:, 0:1, :].expand_as(x_feats)
        x_relative = x_feats - x_repeat
        x_dis = torch.norm(x_relative, dim=-1, keepdim=True)
        x_feats = torch.cat([x_repeat, x_relative, x_dis], dim=-1).view(bsize*num_pts, -1)
        
        x_feats = 2.*math.pi*x_feats @ self.B
        x_feats = torch.cat([torch.sin(x_feats), torch.cos(x_feats)], dim=-1)
        x_feats = self.softmax(self.mlp(x_feats))
        
        permatrix = torch.einsum('bi, ikt->bkt', x_feats, self.permatrix)
        return neigh_indexs, permatrix

    def forward(self, x):
        bsize, num_feat, num_pts = x.size()

        # x0 = get_graph_feature(x, k=self.k)     # (bsize, 3, num_points) -> (bsize, 3*2, num_points, k)
        # t = self.transform_net(x0)              # (bsize, 3, 3)
        # x = x.transpose(2, 1)                   # (bsize, 3, num_points) -> (bsize, num_points, 3)
        # x = torch.bmm(x, t)                     # (bsize, num_points, 3) * (bsize, 3, 3) -> (bsize, num_points, 3)
        # x = x.transpose(2, 1) 
        
        neigh_indexs, permatrix = self.permatrix_lsa(x)
        
        feature = F.gelu(self.conv1(x, neigh_indexs, permatrix))
        x1 = feature.clone()

        feature = F.gelu(self.conv2(feature, neigh_indexs, permatrix))
        x2 = feature.clone()
        
        feature = F.gelu(self.conv3(feature, neigh_indexs, permatrix))
        x3 = feature.clone()

        feature = F.gelu(self.conv4(feature, neigh_indexs, permatrix))
        x4 = feature.clone()

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.gelu(self.conv5(x))
        x1 = F.adaptive_max_pool1d(x, 1).view(bsize, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(bsize, -1)
        x = torch.cat((x1, x2), 1)

        x = F.gelu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = F.gelu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x
