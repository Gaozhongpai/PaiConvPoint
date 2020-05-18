#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# from sparsemax import Sparsemax
# from sinkhorn import Sinkhorn
from pykeops.torch import generic_argkmin

def knn2(K=20):
    knn = generic_argkmin(
        'SqDist(x, y)',
        'a = Vi({})'.format(20),
        'x = Vi({})'.format(3),
        'y = Vj({})'.format(3),
    )
    return knn

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def fibonacci_sphere(samples=1,randomize=False):
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])

    return points

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class ScanConv(nn.Module):
    def __init__(self, in_c, out_c, spiral_size=20, bias=True): # ,device=None):
        super(ScanConv,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        #self.device = device
        self.conv = nn.Linear(in_c*spiral_size,out_c,bias=bias)
        self.adjweight = nn.Parameter(torch.randn(spiral_size, spiral_size), requires_grad=True)
        self.adjweight.data = torch.eye(spiral_size)
        #self.softmax = Sparsemax(dim=-1)  # Sparsemax(dim=-1)#nn.Softmax(dim=1)

    def forward(self, x, spiral_size=20):
        bsize, feats, num_pts = x.size()
        spirals_index = knn(x, spiral_size)
        x = x.permute(0, 2, 1).contiguous().view(bsize*num_pts, feats)

        idx_base = torch.arange(0, bsize, device=x.device).view(-1, 1, 1)*num_pts
        spirals_index = (spirals_index + idx_base).view(-1) # bsize*num_pts*spiral_size
        spirals = x[spirals_index,:].view(bsize*num_pts, spiral_size, feats)
        
        spirals = spirals.permute(0, 2, 1).contiguous()
        spirals = torch.matmul(spirals.view(bsize*num_pts, feats, spiral_size), self.adjweight) 
        spirals = F.elu(spirals.view(bsize*num_pts, feats*spiral_size))
        out_feat = self.conv(spirals).view(bsize,num_pts,self.out_c)  
        out_feat = out_feat.permute(0, 2, 1).contiguous() 
        return out_feat


class DGConv(nn.Module):
    def __init__(self, in_c, out_c, spiral_size=20, bias=True): # ,device=None):
        super(DGConv,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Conv2d(in_c*2, out_c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_c)

    def forward(self,x, spiral_size=20):

        bsize, feats, num_pts = x.size()
        spirals_index = knn(x, spiral_size)
        x = x.permute(0, 2, 1).contiguous().view(bsize*num_pts, feats)

        idx_base = torch.arange(0, bsize, device=x.device).view(-1, 1, 1)*num_pts
        spirals_index = (spirals_index + idx_base).view(-1) # bsize*num_pts*spiral_size
        spirals = x[spirals_index,:].view(bsize, num_pts, spiral_size, feats)

        x = x.view(bsize, num_pts, 1, -1).repeat(1, 1, spiral_size, 1)
        features = torch.cat((spirals - x, x), dim=3) 
        features = features.permute(0, 3, 1, 2).contiguous() 
        out_feat = self.conv(features)  #view(bsize, self.out_c, num_pts, spiral_size)  
        out_feat = self.bn(out_feat.max(dim=-1, keepdim=False)[0]) 

        return out_feat


class DGConv2(nn.Module):
    def __init__(self, in_c, out_c, spiral_size=20, bias=True): # ,device=None):
        super(DGConv2,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv1 = nn.Conv2d(in_c*2, in_c*2, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_c*2, out_c, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(-1)

    def forward(self,x, spiral_size=20):

        bsize, feats, num_pts = x.size()
        spirals_index = knn(x, spiral_size)
        x = x.permute(0, 2, 1).contiguous().view(bsize*num_pts, feats)

        idx_base = torch.arange(0, bsize, device=x.device).view(-1, 1, 1)*num_pts
        spirals_index = (spirals_index + idx_base).view(-1) # bsize*num_pts*spiral_size
        spirals = x[spirals_index,:].view(bsize, num_pts, spiral_size, feats)

        x = x.view(bsize, num_pts, 1, -1).repeat(1, 1, spiral_size, 1)
        features = torch.cat((spirals - x, x), dim=3) 
        features = features.permute(0, 3, 1, 2).contiguous() 
        features = features*self.softmax(self.conv1(features))
        out_feat = self.conv2(features)  #view(bsize, self.out_c, num_pts, spiral_size)  
        out_feat = out_feat.sum(dim=-1)     
        return out_feat


class SpiralConv(nn.Module):
    def __init__(self, in_c, out_c, spiral_size=20, bias=True): # ,device=None):
        super(SpiralConv,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        #self.device = device
        self.conv = nn.Linear(in_c*spiral_size,out_c,bias=bias)
        self.softmax = Sparsemax(dim=-1)  # Sparsemax(dim=-1) #nn.Softmax(dim=1)
        self.kernals = nn.Parameter(torch.rand(in_c, spiral_size) - 0.5, requires_grad=False)
        self.kernals.data[:, 0:1] = torch.zeros(in_c, 1)
        # self.A = nn.Parameter(torch.randn(self.in_c, self.in_c))
        self.one_padding = nn.Parameter(torch.zeros(spiral_size, spiral_size), requires_grad=False)
        self.one_padding.data[0, 0] = 1
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_c)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(self.in_c)

    def forward(self,x, spiral_size=20):
        bsize, feats, num_pts = x.size()
        spirals_index = knn(x, spiral_size)
        x = x.permute(0, 2, 1).contiguous().view(bsize*num_pts, feats)

        idx_base = torch.arange(0, bsize, device=x.device).view(-1, 1, 1)*num_pts
        spirals_index = (spirals_index + idx_base).view(-1) # bsize*num_pts*spiral_size
        spirals = x[spirals_index,:].view(bsize*num_pts, spiral_size, feats)

        # adjweight = torch.matmul(torch.matmul(spirals - spirals[:, 0:1, :], 
        #               (self.A + self.A.transpose(0, 1)) / 2), self.kernals)
        adjweight = torch.matmul(spirals - spirals[:, 0:1, :], self.kernals)
        adjweight = (adjweight + self.one_padding)
        # adjweight = torch.where(adjweight > 0, adjweight, torch.min(adjweight)*5)  # adjweight[adjweight < 0] = torch.min(adjweight)*5
        # adjweight = self.softmax(adjweight.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        adjweight = torch.where(adjweight > 0, adjweight, torch.full_like(adjweight, 0.))  # adjweight[adjweight < 0] = torch.min(adjweight)*5
        adjweight = adjweight / (torch.sum(adjweight, dim=1, keepdim=True) + 1e-6)
        adjweight = adjweight * adjweight
        adjweight = adjweight / (torch.sum(adjweight, dim=1, keepdim=True) + 1e-6)
        adjweight = torch.where(adjweight > 0.1, adjweight, torch.full_like(adjweight, 0.)) 
        
        spirals = spirals.permute(0, 2, 1).contiguous()
        spirals = torch.matmul(spirals, adjweight) 

        spirals = spirals.view(bsize*num_pts, feats*spiral_size)
        out_feat = self.conv(spirals).view(bsize,num_pts,self.out_c)  
        out_feat = out_feat.permute(0, 2, 1).contiguous()      
        return out_feat


class PaiConv(nn.Module):
    def __init__(self, in_c, out_c, kernals, num_neighbor=20, bias=True, is_dim9=False): # ,device=None):
        super(PaiConv,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_neighbor = num_neighbor
        self.kernel_size = kernals.shape[1]
        #self.device = device
        self.mlp = nn.Conv1d(7, in_c, kernel_size=1, bias=bias)
        self.conv = nn.Linear((in_c*2)*self.kernel_size, out_c,bias=bias)
        # self.softmax = Sparsemax(dim=-1)  # Sparsemax(dim=-1) #nn.Softmax(dim=1)
        self.kernals = kernals
        self.one_padding = nn.Parameter(torch.zeros(num_neighbor, self.kernel_size), requires_grad=False)
        self.one_padding.data[0, 0] = 1
        self.bn = nn.BatchNorm1d(out_c)

    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.in_c)
    #     self.A.data.uniform_(-stdv, stdv)
    #     self.A.data += torch.eye(self.in_c)

    def forward(self, x, feature):
        bsize, feats, num_pts = feature.size()
        spirals_index = knn(x, self.num_neighbor)
        x = x.permute(0, 2, 1).contiguous().view(bsize*num_pts, 3)

        idx_base = torch.arange(0, bsize, device=x.device).view(-1, 1, 1)*num_pts
        spirals_index = (spirals_index + idx_base).view(-1) # bsize*num_pts*spiral_size
        
        #### relative position ####
        x_spirals = x[spirals_index,:].view(bsize*num_pts, self.num_neighbor, 3)
        x_relative = x_spirals - x_spirals[:, 0:1, :]
        x_dis = torch.norm(x_relative, dim=-1, keepdim=True)
        x_feats = torch.cat([x_spirals[:, 0:1, :].expand_as(x_spirals), x_relative, x_dis], dim=-1)
        x_feats = self.mlp(x_feats.permute(0, 2, 1).contiguous())
        
        feature = feature.permute(0, 2, 1).contiguous().view(bsize*num_pts, feats)
        spirals = feature[spirals_index,:].view(bsize*num_pts, self.num_neighbor, feats)
        spirals = spirals.permute(0, 2, 1).contiguous()
        spirals = torch.cat([spirals, x_feats], dim=1)

        adjweight = torch.matmul(x_relative, self.kernals)
        adjweight = (adjweight + self.one_padding) #
        adjweight = torch.where(adjweight > 0, adjweight, torch.full_like(adjweight, 0.))  # adjweight[adjweight < 0] = torch.min(adjweight)*5
        # adjweight = self.softmax(adjweight.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        adjweight = adjweight / (torch.sum(adjweight, dim=1, keepdim=True) + 1e-6)
        adjweight = adjweight * adjweight
        adjweight = adjweight / (torch.sum(adjweight, dim=1, keepdim=True) + 1e-6)
        adjweight = torch.where(adjweight > 0.1, adjweight, torch.full_like(adjweight, 0.)) 

        spirals = torch.matmul(spirals, adjweight) 
        spirals = spirals.view(bsize*num_pts, (feats*2)*self.kernel_size)
        out_feat = self.conv(spirals).view(bsize,num_pts,self.out_c)  
        out_feat = self.bn(out_feat.permute(0, 2, 1).contiguous())         
        return out_feat


class TransformIndex(nn.Module):
    def __init__(self, args, kernel_size):
        super(TransformIndex, self).__init__()
        self.k = args.k
        self.kernel_size = kernel_size
        self.kernals = nn.Parameter(torch.rand(3, self.kernel_size-1) - 0.5, requires_grad=True)
        self.kernals.data = torch.tensor(fibonacci_sphere(self.kernel_size-1)).transpose(0, 1)
        self.kernals_padding = nn.Parameter(torch.zeros(3, 1), requires_grad=False)
        # self.softmax = Sparsemax(dim=-1)  # Sparsemax(dim=-1) #nn.Softmax(dim=1)
        # self.A = nn.Parameter(torch.randn(3, 3))
        self.one_padding = nn.Parameter(torch.zeros(self.k, self.kernel_size), requires_grad=False)
        self.one_padding.data[0, 0] = 1
        self.conv = nn.Conv1d(self.k, self.k, kernel_size=1, bias=False)
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(9)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(3)*2

    def forward(self, x):
        batch_size, feats, num_pts = x.size()
        spirals_index = knn(x, self.k)
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_pts
        spirals_index = (spirals_index + idx_base).view(-1) # bsize*num_pts*spiral_size
        spirals = x.permute(0, 2, 1).contiguous().view(batch_size*num_pts, feats)
        spirals = spirals[spirals_index,:].view(batch_size*num_pts, self.k, feats)
        kernals = torch.cat([self.kernals_padding, self.kernals], dim=1)
        #adjweight = torch.matmul(torch.matmul(spirals - spirals[:, 0:1, :], 
        #            (self.A + self.A.transpose(0, 1)) / 2), kernals)
        #### different distance ########
        # adjweight = - torch.norm((spirals - spirals[:, 0:1, :])[:, :, None, :] - 
        #             kernals.transpose(0, 1)[None, None, :, :], dim=3)
        # adjweight = (adjweight - torch.min(adjweight, dim=1, keepdim=True)[0]) / \
        #             (torch.max(adjweight, dim=1, keepdim=True)[0] - torch.min(adjweight, dim=1, keepdim=True)[0])  
        # adjweight = adjweight * adjweight 
        # adjweight = torch.where(adjweight > 0.1, adjweight, torch.full_like(adjweight, 0.))
        # adjweight = adjweight * adjweight              
        # adjweight = torch.where(adjweight > 0.1, adjweight, torch.full_like(adjweight, 0.))
        # adjweight = adjweight / (torch.sum(adjweight, dim=1, keepdim=True) + 1e-6)
        ################################
        adjweight = torch.matmul(spirals - spirals[:, 0:1, :], kernals)
        adjweight = (adjweight + self.one_padding) #
        adjweight = torch.where(adjweight > 0, adjweight, torch.full_like(adjweight, 0.))  # adjweight[adjweight < 0] = torch.min(adjweight)*5
        # adjweight = self.softmax(adjweight.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        adjweight = adjweight / (torch.sum(adjweight, dim=1, keepdim=True) + 1e-6)
        adjweight = adjweight * adjweight
        adjweight = adjweight / (torch.sum(adjweight, dim=1, keepdim=True) + 1e-6)
        adjweight = torch.where(adjweight > 0.1, adjweight, torch.full_like(adjweight, 0.)) 
        return spirals_index, adjweight

class SpiralConvIndex(nn.Module):
    def __init__(self, in_c, out_c, k, kernel_size=20, bias=True): # ,device=None):
        super(SpiralConvIndex,self).__init__()
        self.k = k
        self.kernel_size = kernel_size
        self.in_c = in_c
        self.out_c = out_c
        # self.conv1 = nn.Conv1d(in_c*2, in_c, kernel_size=1, bias=False)
        self.conv = nn.Linear(in_c*self.kernel_size,out_c,bias=bias)
        self.bn = nn.BatchNorm1d(out_c)

    def forward(self, x, spirals_index, adjweight):
        bsize, feats, num_pts = x.size()
        x = x.permute(0, 2, 1).contiguous().view(bsize*num_pts, feats)
        spirals = x[spirals_index,:].view(bsize*num_pts, self.k, feats)
        
        # x = spirals[:, 0:1, :].repeat(1, self.k, 1)
        # spirals = torch.cat([spirals - x, x], dim=2)
        # spirals = self.conv1(spirals.permute(0, 2, 1).contiguous())
        spirals = spirals.permute(0, 2, 1).contiguous()
        
        spirals = torch.matmul(spirals, adjweight)
        spirals = spirals.view(bsize*num_pts, feats*self.kernel_size)

        out_feat = self.conv(spirals).view(bsize,num_pts,self.out_c)  
        out_feat = self.bn(out_feat.permute(0, 2, 1).contiguous())      
        return out_feat

class SpiralConvCombined(nn.Module):
    def __init__(self, in_c, out_c, k, kernel_size=20, bias=True): # ,device=None):
        super(SpiralConvCombined,self).__init__()
        self.k = k
        self.kernel_size = kernel_size
        self.in_c = in_c
        self.out_c = out_c

        self.conv_2 = nn.Conv1d(in_c*self.kernel_size,out_c // 2,kernel_size=1,bias=bias)

        # self.conv_3_1 = nn.Conv2d(in_c*2, in_c*2, kernel_size=1, bias=False)
        # self.conv_3_2 = nn.Conv2d(in_c*2, out_c // 4, kernel_size=1, bias=False)
        # self.softmax = nn.Softmax(-1)

        # self.conv_1 = nn.Conv1d(in_c, out_c // 4, kernel_size=1, bias=False)
        self.conv_4 = nn.Conv2d(in_c*2, out_c // 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_c)


    def forward(self, x, spirals_index, adjweight):
        bsize, feats, num_pts = x.size()
        ### 1 ####
        # out_feat_1 = self.conv_1(x)
        ### 2,3 ####
        spirals = x.permute(0, 2, 1).contiguous().view(bsize*num_pts, feats)
        spirals = spirals[spirals_index,:].view(bsize, num_pts, self.k, feats)
        ## 2 ##
        features = torch.einsum("bnif, bnik -> bfkn", spirals, adjweight) 
        features = features.view(bsize, feats*self.kernel_size, num_pts)
        out_feat_2 = self.conv_2(features)
        ## 3 ##
        x = x[:, :, :, None].repeat(1, 1, 1, self.k)
        spirals = spirals.permute(0, 3, 1, 2).contiguous()
        features = torch.cat((spirals - x, x), dim=1) 
        # out_feat_3 = features*self.softmax(self.conv_3_1(features))
        # out_feat_3 = self.conv_3_2(features).sum(-1)  #view(bsize, self.out_c, num_pts, spiral_size)  
        #### 4 ####
        out_feat_4 = self.conv_4(features).max(dim=-1)[0]
        out_feat = torch.cat([out_feat_2, out_feat_4], dim=1)
        return self.bn(out_feat)


class SpiralConvWO(nn.Module):
    def __init__(self, in_c, out_c, spiral_size=20, bias=True): # ,device=None):
        super(SpiralConvWO,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        #self.device = device
        self.conv = nn.Linear(in_c*spiral_size,out_c,bias=bias)
        self.softmax = Sparsemax(dim=-1)  # Sparsemax(dim=-1) #nn.Softmax(dim=1)
        self.kernals = nn.Parameter(torch.rand(in_c, spiral_size) - 0.5, requires_grad=False)
        self.kernals.data[:, 0:1] = torch.zeros(in_c, 1)
        # self.kernal_padding = nn.Parameter(torch.zeros(in_c, 1), requires_grad=False)
        self.one_padding = nn.Parameter(torch.zeros(spiral_size, spiral_size), requires_grad=False)
        self.one_padding.data[0, 0] = 1

    def forward(self,x, spiral_size=20):
        bsize, feats, num_pts = x.size()
        spirals_index = knn(x, spiral_size)
        x = x.permute(0, 2, 1).contiguous().view(bsize*num_pts, feats)

        idx_base = torch.arange(0, bsize, device=x.device).view(-1, 1, 1)*num_pts
        spirals_index = (spirals_index + idx_base).view(-1) # bsize*num_pts*spiral_size
        spirals = x[spirals_index,:].view(bsize*num_pts, spiral_size*feats)

        out_feat = self.conv(spirals).view(bsize,num_pts,self.out_c)  
        out_feat = out_feat.permute(0, 2, 1).contiguous()      
        return out_feat


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
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 6)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(2, 3))

    def compute_rotation_matrix_from_ortho6d(self, poses):
        x_raw = poses[:, 0:3]  #batch*3
        y_raw = poses[:, 3:6]  #batch*3

        x = F.normalize(x_raw, p=2, dim=1)  #batch*3
        z = F.normalize(torch.cross(x, y_raw), p=2, dim=1)  #batch*3
        y = torch.cross(z, x)  #batch*3
        matrix = torch.stack((x, y, z), dim=2)  #batch*3*3
        return matrix

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = self.compute_rotation_matrix_from_ortho6d(x)
        return x


class PaiDGCNN_v2(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PaiDGCNN_v2, self).__init__()
        self.args = args
        self.k = args.k
        kernel_size = 32
        self.kernals = nn.Parameter(torch.tensor(fibonacci_sphere(kernel_size)).transpose(0, 1), requires_grad=False)
        # self.transform_net = Transform_Net(args)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.conv1 = PaiConv(3, 64, self.kernals, self.k)
        self.conv2 = PaiConv(64, 64, self.kernals, self.k)
        self.conv3 = PaiConv(64, 128, self.kernals, self.k)
        self.conv4 = PaiConv(128, 256, self.kernals, self.k // 2)
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5)
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)
        self.knn = knn2(K=20)

    def pooling(self, x, feature, num_pool):
        batch_size, feats, num_points = x.shape
        x_sub = x[:, :, :num_pool]
        x, feature = x.permute(0, 2, 1).contiguous(), feature.permute(0, 2, 1).contiguous()
        sub_index = self.knn(x_sub.permute(0, 2, 1).contiguous(), x)
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*x.shape[1]
        sub_index = (sub_index + idx_base).view(-1) # bsize*num_pts*spiral_size
        x, feature = x_sub, feature.view(-1, feature.shape[-1])
        feature = feature[sub_index,:].view(batch_size, num_pool, 20, -1)
        feature = torch.max(feature, dim=2)[0].permute(0, 2, 1).contiguous()
        return x, feature

    def forward(self, x):
        batch_size, feats, num_points = x.shape

        # x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        # t = self.transform_net(x0)              # (batch_size, 3, 3)
        # x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        # x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        # x = x.transpose(2, 1) 

        feature = x.clone()
        feature = F.gelu(self.conv1(x, feature))
        #x, feature = x[:, :, :num_points // 4], feature[:, :, :num_points // 4]
        x, feature = self.pooling(x, feature, num_points // 4)
        x1 = feature[:, :, :num_points // 32].clone() 

        feature = F.gelu(self.conv2(x, feature))
        #x, feature = x[:, :, :num_points // 16], feature[:, :, :num_points // 16]
        x, feature = self.pooling(x, feature, num_points // 8)    
        x2 = feature[:, :, :num_points // 32].clone() 
        
        feature = F.gelu(self.conv3(x, feature))
        #x, feature = x[:, :, :num_points // 32], feature[:, :, :num_points // 32]
        x, feature = self.pooling(x, feature, num_points // 16)
        x3 = feature[:, :, :num_points // 32].clone() 

        feature = F.gelu(self.conv4(x, feature))
        #x, feature = x[:, :, :num_points // 64], feature[:, :, :num_points // 64]
        x, feature = self.pooling(x, feature, num_points // 32)
        x4 = feature[:, :, :num_points // 32].clone() 

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.gelu(self.conv5(x))
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.gelu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = F.gelu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class PaiDGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PaiDGCNN, self).__init__()
        self.args = args
        self.k = args.k
        num_kernel = 32
        self.transform = TransformIndex(args, kernel_size=num_kernel)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.conv1 = SpiralConvIndex(3, 64, self.k, num_kernel)
        self.conv2 = SpiralConvIndex(64, 64, self.k, num_kernel)
        self.conv3 = SpiralConvIndex(64, 128, self.k, num_kernel)
        self.conv4 = SpiralConvIndex(128, 256, self.k, num_kernel)
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
        batch_size = x.size(0)

        # x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        # t = self.transform_net(x0)              # (batch_size, 3, 3)
        # x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        # x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        # x = x.transpose(2, 1) 

        spirals_index, adjweight = self.transform(x) 
        x = F.gelu(self.conv1(x, spirals_index, adjweight))
        x1 = x.clone()

        x = F.gelu(self.conv2(x, spirals_index, adjweight))
        x2 = x.clone()
        
        x = F.gelu(self.conv3(x, spirals_index, adjweight))
        x3 = x.clone()

        x = F.gelu(self.conv4(x, spirals_index, adjweight))
        x4 = x.clone()

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.gelu(self.conv5(x))
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.gelu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = F.gelu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class PaiDGCNN2(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PaiDGCNN2, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.conv1 = nn.Sequential(DGConv2(3, 64), self.bn1)
        self.conv2 = nn.Sequential(DGConv2(64, 64), self.bn2)
        self.conv3 = nn.Sequential(DGConv2(64, 128), self.bn3)
        self.conv4 = nn.Sequential(DGConv2(128, 256), self.bn4)
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5)
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)
        

    def forward(self, x):
        batch_size = x.size(0)

        spirals_index = knn(x, self.k)
        x = F.gelu(self.conv1(x))
        x1 = x.clone()

        x = F.gelu(self.conv2(x))
        x2 = x.clone()
        
        x = F.gelu(self.conv3(x))
        x3 = x.clone()

        x = F.gelu(self.conv4(x))
        x4 = x.clone()

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.gelu(self.conv5(x))
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
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
        batch_size = x.size(0)
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
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
