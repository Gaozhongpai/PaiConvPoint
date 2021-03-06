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
from util import fibonacci_sphere, knn, knn3, topkmax
import math
from pointnet2_ops import pointnet2_utils
from SoftPool import soft_pool2d

class PaiConv(nn.Module):
    def __init__(self, in_c, out_c, kernels, k, num_neighbor, bias=True): # ,device=None):
        super(PaiConv, self).__init__()
        self.k = k
        self.num_neighbor = num_neighbor
        self.in_c = in_c
        self.out_c = out_c
        self.knn = knn3(self.k)
        self.kernels = kernels
        self.A = nn.Parameter(torch.randn(3, 3))
        self.reset_parameters()

        self.map_size = 32
        self.group = 4
        self.B = nn.Parameter(torch.randn(7, self.map_size) , requires_grad=False)
        self.one_padding = nn.Parameter(torch.zeros(self.k, self.num_neighbor), requires_grad=False)
        self.one_padding.data[0, 0] = 1

        self.in_c_x = in_c // 2 if in_c > 3 else in_c
        self.mlp = nn.Conv1d(self.map_size*2, self.in_c_x, kernel_size=1, bias=bias)
        self.conv1 = nn.Conv2d((in_c+self.in_c_x)*2,out_c,kernel_size=1,bias=bias)
        # self.conv2 = nn.Linear((in_c+self.in_c_x)*self.num_neighbor,out_c,bias=bias)
        self.bn = nn.BatchNorm1d(out_c)
        self.mlp_out = nn.Conv1d(in_c, out_c, kernel_size=1, bias=False)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(9)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(3)*2

    def forward(self, x, feature):
        bsize, num_feat, num_pts = feature.size()
        x = x.permute(0, 2, 1).contiguous()
        neigh_index = self.knn(x, x)
        x = x.view(bsize*num_pts, 3)

        idx_base = torch.arange(0, bsize, device=x.device).view(-1, 1, 1)*num_pts
        neigh_index = (neigh_index + idx_base).view(-1) # bsize*num_pts*spiral_size

        #### relative position ####
        x_neighs = x[neigh_index,:].view(bsize*num_pts, self.k, 3)
        x_repeat = x_neighs[:, 0:1, :].expand_as(x_neighs)
        x_relative = x_neighs - x_repeat
        x_dis = torch.norm(x_relative, dim=-1, keepdim=True)
        x_feats = torch.cat([x_repeat, x_relative, x_dis], dim=-1)
        x_feats = 2.*math.pi * (x_feats-x_feats.min()/(x_feats.max()-x_feats.min())) @ self.B
        x_feats = torch.cat([torch.sin(x_feats), torch.cos(x_feats)], dim=-1)
        x_feats = self.mlp(x_feats.permute(0, 2, 1).contiguous())

        permatrix = torch.matmul(x_relative, (self.A + self.A.t())/2 @ self.kernels)
        permatrix = (permatrix + self.one_padding) #
        permatrix = topkmax(permatrix)

        ##### 1 #####
        feats = feature.permute(0, 2, 1).contiguous().view(bsize*num_pts, num_feat)
        feats = feats[neigh_index,:].view(bsize*num_pts, self.k, num_feat)
        feats = feats.permute(0, 2, 1).contiguous()
        feats = torch.cat([feats, x_feats], dim=1)

        f_repeat =  feats[:, :, 0:1].expand_as(feats)
        f_relative = feats - f_repeat
        # f_dis = torch.norm(f_relative, dim=1, keepdim=True)
        feats = torch.cat([f_repeat, f_relative], dim=1)
        num_feat = (num_feat + self.in_c_x)*2
        

        if num_feat > 3 + self.in_c_x: ## channel shuffle
            feats = feats.view(bsize*num_pts,self.group, num_feat//self.group,-1).permute(0,2,1,3).reshape(bsize*num_pts, num_feat,-1)
        
        feats = torch.matmul(feats, permatrix / (torch.sum(permatrix, dim=1, keepdim=True) + 1e-6)) 
        feats = feats.view(bsize, num_pts, num_feat, self.num_neighbor).permute(0, 2, 1, 3)
        out_feat = F.gelu(torch.max(self.conv1(feats), dim=-1)[0])
        
        ##### 2 #####
        # feats = feats.permute(0, 2, 1).contiguous().view(bsize*num_pts, -1)
        # feats = feats[neigh_index,:].view(bsize*num_pts, self.k, -1)
        # feats = feats.permute(0, 2, 1).contiguous()
        # feats = torch.matmul(feats, permatrix).view(bsize*num_pts, -1) 

        # out_feat = self.conv2(feats).view(bsize,num_pts,self.out_c).permute(0, 2, 1).contiguous()  
        out_feat = self.bn(out_feat + self.mlp_out(feature))     
        return out_feat

class PaiPool(nn.Module):
    def __init__(self, in_c, k, nlayer, bias=False): # ,device=None):
        super(PaiPool, self).__init__()
        # self.knn = knn3(self.k)
        self.k = k
        self.nlayer = nlayer
        self.mlp = nn.Conv2d(in_c, in_c, kernel_size=1, bias=bias)
        
    def forward(self, x, feature, num_pool):
        x = x.contiguous()
        xyz_flipped = x.transpose(1, 2).contiguous()
        x_sub = (
            pointnet2_utils.gather_operation(
                x, pointnet2_utils.furthest_point_sample(xyz_flipped, num_pool)
            ).transpose(1, 2).contiguous()
        )
        # sub_index = self.knn(x_sub, xyz_flipped).int()
        sub_index = pointnet2_utils.ball_query(0.2*self.nlayer, self.k, xyz_flipped, x_sub)
        x = pointnet2_utils.grouping_operation(x, sub_index)
        x = torch.max(x, dim=-1)[0]
        # x = soft_pool2d(x, [1, x.shape[-1]]).squeeze(-1)
        feature = pointnet2_utils.grouping_operation(feature, sub_index)
        feature = self.mlp(feature)
        feature = torch.max(feature, dim=-1)[0]
        # feature = soft_pool2d(feature, [1, feature.shape[-1]]).squeeze(-1)
        return x, feature 
    
class PaiNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PaiNet, self).__init__()
        self.args = args
        self.k = args.k
        num_kernel = 9
        self.num_layers = 4
        self.k_down = 5

        
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.kernals = nn.Parameter(torch.tensor(fibonacci_sphere(num_kernel)).transpose(0, 1), requires_grad=False)

        self.conv1 = PaiConv(3, 64, self.kernals, self.k, num_kernel)
        self.conv2 = PaiConv(64, 64, self.kernals, self.k, num_kernel)
        self.conv3 = PaiConv(64, 128, self.kernals, self.k, num_kernel)
        self.conv4 = PaiConv(128, 256, self.kernals, self.k, num_kernel)
        self.conv6 = PaiConv(256, 512, self.kernals, self.k, num_kernel)

        self.pool1 = PaiPool(64, self.k_down, 1)
        self.pool2 = PaiPool(64, self.k_down, 1)
        self.pool3 = PaiPool(128, self.k_down, 1)
        self.pool4 = PaiPool(256, self.k_down, 1)

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
        
        feature = F.gelu(self.conv1(x, x))
        x, feature = self.pool1(x, feature, num_pts // 4)
        # x1 = feature[:, :, :num_pts // 32]

        feature = F.gelu(self.conv2(x, feature))
        x, feature = self.pool2(x, feature, num_pts // 8)
        # x2 = feature[:, :, :num_pts // 32]
        
        feature = F.gelu(self.conv3(x, feature))
        x, feature = self.pool3(x, feature, num_pts // 16)
        # x3 = feature[:, :, :num_pts // 32]

        feature = F.gelu(self.conv4(x, feature))
        x, feature = self.pool4(x, feature, num_pts // 32)

        feature = F.gelu(self.conv6(x, feature))
        # _, x4 = self.pooling(x, feature, num_pts // 32)

        # x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.gelu(self.conv5(feature))
        x1 = F.adaptive_max_pool1d(x, 1).view(bsize, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(bsize, -1)
        x = torch.cat((x1, x2), 1)

        x = F.gelu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = F.gelu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x


