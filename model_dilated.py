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
import torch.nn.init as init
import torch.nn.functional as F
from util import fibonacci_sphere, knn, knn3, topkmax, compute_rotation_matrix_from_ortho6d
import math
from sparsemax import Sparsemax
from cbam import BasicConv


class PaiConv(nn.Module):
    def __init__(self, in_c, out_c, kernels, k, num_kernel, dilation=1, bias=True): # ,device=None):
        super(PaiConv, self).__init__()
        self.k = k
        self.num_kernel = num_kernel
        self.in_c = in_c
        self.out_c = out_c
        self.dilation = dilation
        self.kernels = kernels
        self.one_padding = nn.Parameter(torch.zeros(self.k, self.num_kernel), requires_grad=False)
        self.one_padding.data[0, 0] = 1

        # self.map_size = 32
        # self.B = nn.Parameter(torch.randn(7, self.map_size) , requires_grad=False)
        # self.mlptmp1 = BasicConv(self.map_size*2, self.map_size*2, kernel_size=1)
        # self.mlptmp2 = nn.Conv1d(self.map_size*2, 1, kernel_size=1, bias=False)
        # self.temp_factor = 100

        self.conv = BasicConv(in_c*self.num_kernel,out_c, kernel_size=1)
        # self.mlp_out = BasicConv(in_c, out_c, kernel_size=1)

    def forward(self, x, feature, neigh_indexs):
        bsize, num_feat, num_pts = feature.size()
        x = x.permute(0, 2, 1).contiguous()
        neigh_index = neigh_indexs[:, :, :self.k*self.dilation:self.dilation]
        x = x.view(bsize*num_pts, 3)

        idx_base = torch.arange(0, bsize, device=x.device).view(-1, 1, 1)*num_pts
        neigh_index = (neigh_index + idx_base).view(-1) # bsize*num_pts*spiral_size

        #### relative position ####
        x_neighs = x[neigh_index,:].view(bsize*num_pts, self.k, 3)
        x_repeat = x_neighs[:, 0:1, :].expand_as(x_neighs)
        x_relative = x_neighs - x_repeat

        # x_dis = torch.norm(x_relative, dim=-1, keepdim=True)
        # x_feats = torch.cat([x_repeat, x_relative, x_dis], dim=-1)
        # x_feats = 2.*math.pi * (x_feats-x_feats.min()/(x_feats.max()-x_feats.min())) @ self.B
        # x_feats = torch.cat([torch.sin(x_feats), torch.cos(x_feats)], dim=-1).permute(0, 2, 1).contiguous()
        # tmpt = self.mlptmp2(self.mlptmp1(x_feats.unsqueeze(-1)).squeeze(-1)).permute(0, 2, 1).contiguous()
        # tmpt = torch.sigmoid(tmpt)*(0.1 - 1.0/self.temp_factor) + 1.0/self.temp_factor 

        feats = feature.permute(0, 2, 1).contiguous().view(bsize*num_pts, num_feat)
        feats = feats[neigh_index,:].view(bsize*num_pts, self.k, num_feat)
        feats = feats.permute(0, 2, 1).contiguous()
        
        permatrix = torch.matmul(x_relative, self.kernels) 
        permatrix = (permatrix + self.one_padding) #
        permatrix = topkmax(permatrix) #/tmpt)

        feats = torch.matmul(feats, permatrix).view(bsize, num_pts, -1, 1)
        feats = feats.permute(0, 2, 1, 3).contiguous()
        out_feat = self.conv(feats).squeeze(-1) + feature if self.in_c == self.out_c else self.conv(feats).squeeze(-1)
        return out_feat


class PaiConvDG(nn.Module):
    def __init__(self, in_c, out_c, kernels, k, num_kernel, dilation=1, bias=True): # ,device=None):
        super(PaiConvDG, self).__init__()
        self.k = k
        self.num_kernel = num_kernel
        self.in_c = in_c
        self.out_c = out_c
        self.dilation = dilation
        self.kernels = kernels
        self.conv = nn.Conv1d(2*in_c,out_c,kernel_size=1)
        self.bn = nn.BatchNorm1d(out_c)
        self.one_padding = nn.Parameter(torch.zeros(self.k, self.num_kernel), requires_grad=False)
        self.one_padding.data[0, 0] = 1
        self.group = 4

    def forward(self, x, feature, neigh_indexs):
        bsize, num_feat, num_pts = feature.size()
        x = x.permute(0, 2, 1).contiguous()
        neigh_index = neigh_indexs[:, :, :self.k*self.dilation:self.dilation]
        x = x.view(bsize*num_pts, 3)

        idx_base = torch.arange(0, bsize, device=x.device).view(-1, 1, 1)*num_pts
        neigh_index = (neigh_index + idx_base).view(-1) # bsize*num_pts*spiral_size

        #### relative position ####
        x_neighs = x[neigh_index,:].view(bsize*num_pts, self.k, 3)
        x_repeat = x_neighs[:, 0:1, :].expand_as(x_neighs)
        x_relative = x_neighs - x_repeat

        feats = feature.permute(0, 2, 1).contiguous().view(bsize*num_pts, num_feat)
        feats = feats[neigh_index,:].view(bsize*num_pts, self.k, num_feat)
        feats = feats.permute(0, 2, 1).contiguous()

        f_repeat =  feats[:, :, 0:1].expand_as(feats)
        f_relative = feats - f_repeat
        # f_dis = torch.norm(f_relative, dim=1, keepdim=True)
        feats = torch.cat([f_repeat, f_relative], dim=1)

        permatrix = torch.matmul(x_relative, self.kernels) 
        permatrix = (permatrix + self.one_padding) #
        permatrix = topkmax(permatrix) #/tmpt)

        num_feat = num_feat*2
        if num_feat > 2*3: ## channel shuffle
            feats = feats.view(bsize*num_pts,self.group, num_feat//self.group,-1).permute(0,2,1,3).reshape(bsize*num_pts, num_feat,-1)
        feats = torch.matmul(feats, permatrix / (torch.sum(permatrix, dim=1, keepdim=True) + 1e-6))
        feats = feats.view(bsize*num_pts, num_feat, self.num_kernel)

        out_feat = torch.max(self.conv(feats), dim=-1)[0]
        out_feat = self.bn(out_feat.view(bsize, num_pts,-1).permute(0, 2, 1).contiguous())      
        out_feat = out_feat + feature if self.in_c == self.out_c else out_feat
        return out_feat

class PaiNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PaiNet, self).__init__()
        self.args = args
        self.k = args.k
        num_kernel = 12
        self.num_layers = 6
        self.knn = knn3(self.k*self.num_layers)

        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.kernals = nn.Parameter(torch.tensor(fibonacci_sphere(num_kernel)).transpose(0, 1), requires_grad=False)

        self.conv1 = PaiConvDG(3, 64, self.kernals, self.k, num_kernel, 1)
        self.conv2 = PaiConvDG(64, 64, self.kernals, self.k, num_kernel, 2)
        self.conv3 = PaiConvDG(64, 64, self.kernals, self.k, num_kernel, 3)
        self.conv4 = PaiConvDG(64, 64, self.kernals, self.k, num_kernel, 4)
        self.conv4a = PaiConvDG(64, 128, self.kernals, self.k, num_kernel, 5)
        self.conv4b = PaiConvDG(128, 128, self.kernals, self.k, num_kernel, 6)

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
        # self.transform_net = Transform_Net(args)

    def forward(self, x):
        bsize, feats, num_pts = x.size()

        # x0 = get_graph_feature(x, k=self.k)     # (bsize, 3, num_points) -> (bsize, 3*2, num_points, k)
        # t = self.transform_net(x0)              # (bsize, 3, 3)
        # x = x.transpose(2, 1)                   # (bsize, 3, num_points) -> (bsize, num_points, 3)
        # x = torch.bmm(x, t)                     # (bsize, num_points, 3) * (bsize, 3, 3) -> (bsize, num_points, 3)
        # x = x.transpose(2, 1) 

        neigh_indexs = knn(x, self.k*self.num_layers)
        feature = self.conv1(x, x, neigh_indexs)
        x1 = feature.clone()

        feature = self.conv2(x, feature, neigh_indexs)
        x2 = feature.clone()
        
        feature = self.conv3(x, feature, neigh_indexs)
        x3 = feature.clone()

        feature = self.conv4(x, feature, neigh_indexs)
        x4 = feature.clone()
        feature = self.conv4a(x, feature, neigh_indexs)
        x4a = feature.clone()
        feature = self.conv4b(x, feature, neigh_indexs)
        x4b = feature.clone()

        x = torch.cat((x1, x2, x3, x4, x4a, x4b), dim=1)
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
