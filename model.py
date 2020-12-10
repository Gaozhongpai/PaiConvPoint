#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

from sparsemax import Sparsemax
import torch
import torch.nn as nn
import torch.nn.functional as F
# from sparsemax import Sparsemax
# from sinkhorn import Sinkhorn
from util import fibonacci_sphere, knn, topkmax
    
class PaiConv(nn.Module):
    def __init__(self, in_c, out_c, k, kernel_size=20, bias=True): # ,device=None):
        super(PaiConv, self).__init__()
        self.k = k
        self.kernel_size = kernel_size
        self.in_c = in_c
        self.out_c = out_c
        self.group = 4
        self.conv = nn.Linear(in_c*self.kernel_size,out_c,bias=bias)
        self.bn = nn.BatchNorm1d(out_c)

    def forward(self, feature, neigh_indexs, permatrix):
        bsize, num_feat, num_pts = feature.size()
        feature = feature.permute(0, 2, 1).contiguous().view(bsize*num_pts, num_feat)
        
        feats = feature[neigh_indexs,:].view(bsize*num_pts, self.k, num_feat)
        feats = feats.permute(0, 2, 1).contiguous()
        # x_feat = self.mlp(x_feat.permute(0, 2, 1).contiguous())
        # feats = torch.cat([x_feat, feats], dim=1)
        
        if num_feat > 3: ## channel shuffle
            feats = feats.view(bsize*num_pts,self.group, num_feat//self.group,-1).permute(0,2,1,3).reshape(bsize*num_pts, num_feat,-1)
        feats = torch.matmul(feats, permatrix)

        feats = feats.view(bsize*num_pts, num_feat*self.kernel_size)
        out_feat = self.conv(feats).view(bsize,num_pts,self.out_c)  
        out_feat = self.bn(out_feat.permute(0, 2, 1).contiguous())      
        return out_feat

class PaiConvMax(nn.Module):
    def __init__(self, in_c, out_c, k, kernel_size=20, bias=True): # ,device=None):
        super(PaiConvMax, self).__init__()
        self.k = k
        self.kernel_size = kernel_size
        self.in_c = in_c
        self.out_c = out_c
        self.group = 4
        self.conv1 = nn.Linear(in_c*self.kernel_size,out_c//2,bias=bias)
        self.conv2 = nn.Conv1d(in_c*2,out_c//2,kernel_size=1,bias=bias)
        self.bn = nn.BatchNorm1d(out_c)

    def forward(self, feature, neigh_indexs, permatrix):
        bsize, num_feat, num_pts = feature.size()
        feature = feature.permute(0, 2, 1).contiguous().view(bsize*num_pts, num_feat)
        
        feats = feature[neigh_indexs,:].view(bsize*num_pts, self.k, num_feat)
        feats = feats.permute(0, 2, 1).contiguous()
        # x_feat = self.mlp(x_feat.permute(0, 2, 1).contiguous())
        # feats = torch.cat([x_feat, feats], dim=1)

        if num_feat > 3: ## channel shuffle
            feats = feats.view(bsize*num_pts,self.group, num_feat//self.group,-1).permute(0,2,1,3).reshape(bsize*num_pts, num_feat,-1)
        feats = torch.matmul(feats, permatrix)

        feats = feats.view(bsize*num_pts, num_feat*self.kernel_size)
        out_feat1 = self.conv1(feats).view(bsize,num_pts,self.out_c//2)  
        feats = feats.view(bsize*num_pts, num_feat, self.kernel_size)
        f_repeat =  feats[:, :, 0:1].expand_as(feats)
        feats = torch.cat([f_repeat, feats - f_repeat], dim=1)
        out_feat2 = torch.max(self.conv2(feats), dim=-1)[0].view(bsize,num_pts,self.out_c//2)  
        out_feat = torch.cat([out_feat1, out_feat2], dim=-1)
        out_feat = self.bn(out_feat.permute(0, 2, 1).contiguous())      
        return out_feat

class PaiConvDG(nn.Module):
    def __init__(self, in_c, out_c, k, kernel_size=20, bias=True): # ,device=None):
        super(PaiConvDG, self).__init__()
        self.k = k
        self.kernel_size = kernel_size
        self.in_c = in_c
        self.out_c = out_c
        self.group = 4
        self.conv = nn.Conv1d(2*in_c,out_c,kernel_size=1,bias=bias)
        self.bn = nn.BatchNorm1d(out_c)

    def forward(self, feature, neigh_indexs, permatrix):
        bsize, num_feat, num_pts = feature.size()
        feature = feature.permute(0, 2, 1).contiguous().view(bsize*num_pts, num_feat)
        
        feats = feature[neigh_indexs,:].view(bsize*num_pts, self.k, num_feat)
        feats = feats.permute(0, 2, 1).contiguous()
        f_repeat =  feats[:, :, 0:1].expand_as(feats)
        f_relative = feats - f_repeat
        # f_dis = torch.norm(f_relative, dim=1, keepdim=True)
        feats = torch.cat([f_repeat, f_relative], dim=1)

        num_feat = num_feat*2
        # if num_feat > 2*3: ## channel shuffle
        #     feats = feats.view(bsize*num_pts,self.group, num_feat//self.group,-1).permute(0,2,1,3).reshape(bsize*num_pts, num_feat,-1)
        feats = torch.matmul(feats, permatrix)
        feats = feats.view(bsize*num_pts, num_feat, self.kernel_size)

        out_feat = torch.max(self.conv(feats), dim=-1)[0].view(bsize,num_pts,self.out_c)  
        out_feat = self.bn(out_feat.permute(0, 2, 1).contiguous())      
        return out_feat
    
class PaiNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PaiNet, self).__init__()
        self.args = args
        self.k = args.k
        num_kernel = 9 # xyz*2+1 # args.k
        
        self.kernels = nn.Parameter(torch.tensor(fibonacci_sphere(num_kernel)).transpose(0, 1), requires_grad=False)        
        self.one_padding = nn.Parameter(torch.zeros(self.k, num_kernel), requires_grad=False)
        self.one_padding.data[0, 0] = 1
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = Sparsemax(dim=-1)

        self.conv1 = PaiConvDG(3, 64, self.k, num_kernel)
        self.conv2 = PaiConvDG(64, 64, self.k, num_kernel)
        self.conv3 = PaiConvDG(64, 128, self.k, num_kernel)
        self.conv4 = PaiConvDG(128, 256, self.k, num_kernel)

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

    def permatrix_best(self, x):
        bsize, num_feat, num_pts = x.size()
        neigh_indexs = knn(x, self.k)
        idx_base = torch.arange(0, bsize, device=x.device).view(-1, 1, 1)*num_pts
        neigh_indexs = (neigh_indexs + idx_base).view(-1) # bsize*num_pts*spiral_size

        x_feats = x.permute(0, 2, 1).contiguous().view(bsize*num_pts, -1)
        x_feats = x_feats[neigh_indexs,:].view(bsize*num_pts, self.k, -1)
        x_repeat = x_feats[:, 0:1, :].expand_as(x_feats)
        x_relative = x_feats - x_repeat

        permatrix = torch.matmul(x_relative, self.kernels)
        permatrix = (permatrix + self.one_padding) #
        permatrix = topkmax(permatrix) # self.softmax(permatrix.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous() # 
        return neigh_indexs, permatrix

    def forward(self, x):
        bsize, num_feat, num_pts = x.size()

        # x0 = get_graph_feature(x, k=self.k)     # (bsize, 3, num_points) -> (bsize, 3*2, num_points, k)
        # t = self.transform_net(x0)              # (bsize, 3, 3)
        # x = x.transpose(2, 1)                   # (bsize, 3, num_points) -> (bsize, num_points, 3)
        # x = torch.bmm(x, t)                     # (bsize, num_points, 3) * (bsize, 3, 3) -> (bsize, num_points, 3)
        # x = x.transpose(2, 1) 

        neigh_indexs, permatrix = self.permatrix_best(x)
                
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


