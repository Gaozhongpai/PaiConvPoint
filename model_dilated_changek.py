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
from networks import get_graph_feature
import math

class RandLANet(nn.Module):
    def __init__(self, in_c, out_c, k, kernel_size=20, bias=True): # ,device=None):
        super(RandLANet, self).__init__()
        self.k = k
        self.kernel_size = kernel_size
        self.in_c = in_c
        self.out_c = out_c
        self.mlp = nn.Conv1d(in_c, in_c, kernel_size=1, bias=False)
        self.conv = nn.Linear(in_c,out_c,bias=bias)
        self.bn = nn.BatchNorm1d(out_c)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature, neigh_index, permatrix):
        bsize, feats, num_pts = feature.size()
        feature = feature.permute(0, 2, 1).contiguous().view(bsize*num_pts, feats)
        
        spirals = feature[neigh_index,:].view(bsize*num_pts, self.k, feats)
        spirals = spirals.permute(0, 2, 1).contiguous()
        
        spirals = torch.sum(self.softmax(self.mlp(spirals))*spirals, dim=-1) ## This is for RandLA-Net
        spirals = spirals.view(bsize*num_pts, feats)

        out_feat = self.conv(spirals).view(bsize,num_pts,self.out_c)  
        out_feat = self.bn(out_feat.permute(0, 2, 1).contiguous())      
        return out_feat
    
class PaiConv(nn.Module):
    def __init__(self, in_c, out_c, k, num_neighbor, dilation=1, bias=True): # ,device=None):
        super(PaiConv, self).__init__()
        self.k = k
        self.num_neighbor = math.ceil(k / dilation)
        self.in_c = in_c
        self.out_c = out_c
        self.dilation = dilation
        self.map_size = 32
        self.group = 4
        self.B = nn.Parameter(torch.randn(7, self.map_size) , requires_grad=False)  
        self.kernels = nn.Parameter(torch.tensor(fibonacci_sphere(self.num_neighbor)).transpose(0, 1), requires_grad=False)
        self.one_padding = nn.Parameter(torch.zeros(self.num_neighbor, self.num_neighbor), requires_grad=False)
        self.one_padding.data[0, 0] = 1
 
        self.in_c_x = in_c // 2 if in_c > 3 else in_c
        self.mlp = nn.Conv1d(self.map_size*2, self.in_c_x, kernel_size=1, bias=bias)
        self.conv = nn.Linear((in_c+self.in_c_x)*self.num_neighbor, out_c,bias=bias)
        
        # self.mlp_out = nn.Conv1d(in_c, out_c, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm1d(out_c)

    def forward(self, x, feature, neigh_indexs):
        bsize, num_feat, num_pts = feature.size()
        x = x.permute(0, 2, 1).contiguous()
        neigh_index = neigh_indexs[:, :, :self.k:self.dilation]
        x = x.view(bsize*num_pts, 3)

        idx_base = torch.arange(0, bsize, device=x.device).view(-1, 1, 1)*num_pts
        neigh_index = (neigh_index + idx_base).view(-1) # bsize*num_pts*spiral_size

        #### relative position ####
        x_neighs = x[neigh_index,:].view(bsize*num_pts, self.num_neighbor, 3)
        x_repeat = x_neighs[:, 0:1, :].expand_as(x_neighs)
        x_relative = x_neighs - x_repeat
        x_dis = torch.norm(x_relative, dim=-1, keepdim=True)
        x_feats = 2.*math.pi*torch.cat([x_repeat, x_relative, x_dis], dim=-1) @ self.B
        x_feats = torch.cat([torch.sin(x_feats), torch.cos(x_feats)], dim=-1)
        x_feats = self.mlp(x_feats.permute(0, 2, 1).contiguous())

        feats = feature.permute(0, 2, 1).contiguous().view(bsize*num_pts, num_feat)
        feats = feats[neigh_index,:].view(bsize*num_pts, self.num_neighbor, num_feat)
        feats = feats.permute(0, 2, 1).contiguous()
        feats = torch.cat([feats, x_feats], dim=1)
        num_feat = num_feat + self.in_c_x
        
        permatrix = torch.matmul(x_relative, self.kernels)
        permatrix = (permatrix + self.one_padding) #
        permatrix = topkmax(permatrix)

        # if num_feat > 2*3: ## channel shuffle
        #     feats = feats.view(bsize*num_pts,self.group, num_feat//self.group,-1).permute(0,2,1,3).reshape(bsize*num_pts, num_feat,-1)
        feats = torch.matmul(feats, permatrix) 
        feats = feats.view(bsize*num_pts, num_feat*self.num_neighbor)
        out_feat = self.conv(feats).view(bsize,num_pts,self.out_c)  
        
        out_feat = self.bn(out_feat.permute(0, 2, 1).contiguous()) # + self.mlp_out(feature)
        return out_feat


class PaiNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PaiNet, self).__init__()
        self.args = args
        self.k = args.k
        num_kernel = 7
        self.knn = knn3(self.k)

        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.conv1 = PaiConv(3, 64, self.k, num_kernel, 1)
        self.conv2 = PaiConv(64, 64, self.k, num_kernel, 2)
        self.conv3 = PaiConv(64, 128, self.k, num_kernel, 3)
        self.conv4 = PaiConv(128, 256, self.k, num_kernel, 4)

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
        
        x_temp = x.permute(0, 2, 1).contiguous()
        neigh_indexs = self.knn(x_temp, x_temp)
        
        feature = F.gelu(self.conv1(x, x, neigh_indexs))
        x1 = feature.clone()

        feature = F.gelu(self.conv2(x, feature, neigh_indexs))
        x2 = feature.clone()
        
        feature = F.gelu(self.conv3(x, feature, neigh_indexs))
        x3 = feature.clone()

        feature = F.gelu(self.conv4(x, feature, neigh_indexs))
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


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        bsize = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(bsize, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(bsize, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x1 = F.adaptive_max_pool1d(x, 1).squeeze()
        x2 = F.adaptive_avg_pool1d(x, 1).squeeze()
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x