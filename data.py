#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import open3d as o3d
import time
import torch

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = self.data[item][:self.num_points]
            # pointcloud = self.data[item][np.random.permutation(self.data.shape[1])[:self.num_points]]
            pointcloud = translate_pointcloud(pointcloud)
            pointcloud = jitter_pointcloud(pointcloud)
            # np.random.shuffle(pointcloud)
        else:
            pointcloud = self.data[item][:self.num_points]
            # pointcloud = self.data[item][np.random.permutation(self.data.shape[1])[:self.num_points]]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

def translate_pointcloud_tensor(pointcloud): #, center):
    xyz1 = torch.empty(3).uniform_(2./3., 3./2.)
    xyz2 = torch.empty(3).uniform_(-0.2, 0.2)
    translated_pointcloud = pointcloud*xyz1 + xyz2
    # center = [sub_center*xyz1 + xyz2 for sub_center in center]
    return translated_pointcloud # , center

def jitter_pointcloud_tensor(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += torch.clamp(sigma * torch.randn(N, C), -1*clip, clip)
    return pointcloud

def point_loader_train(input):
    #data = np.loadtxt(input, delimiter=',', dtype=np.float32)

    mesh = torch.load(input)
    data = mesh['data']
    # data, center = mesh['data'], mesh['center']
    # data = data - (torch.max(data, 0)[0] + torch.min(data, 0)[0]) / 2.
    # data = 2 * data / (torch.max(data) - torch.min(data))
    #noise = torch.empty_like(data).normal_(0, 0.001)
    #data = data + noise

    pointcloud = data[torch.randperm(data.shape[0])[:8192]]
    pointcloud = translate_pointcloud_tensor(pointcloud)
    pointcloud = jitter_pointcloud_tensor(pointcloud)
    # np.random.shuffle(pointcloud)

    return pointcloud

def point_loader_test(input):
    #data = np.loadtxt(input, delimiter=',', dtype=np.float32)

    mesh = torch.load(input)
    data = mesh['data']
    # pointcloud = data[:8196]
    pointcloud = data[torch.randperm(data.shape[0])[:8192]]
    return pointcloud

if __name__ == '__main__':
    train = ModelNet40(512)
    test = ModelNet40(512, 'test')
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    pcd = o3d.geometry.PointCloud()
    for i, (data, label) in enumerate(train):
        if i < 20:            
            pcd.points = o3d.utility.Vector3dVector(data)
            vis.add_geometry(pcd)

            ctr = vis.get_view_control()
            ctr.set_front([1,1,-1])
            ctr.set_up([0,1,0])
            ctr.set_zoom(0.9)

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(1)
            vis.capture_screen_image('{}b.png'.format(i))
            vis.remove_geometry(pcd)

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(data)
            # o3d.visualization.draw_geometries([pcd],
            #                         zoom=1,
            #                         front=[1, 1, -1],
            #                         lookat=[0, 0, 0],
            #                         up=[0, 1, 0])
            print(data.shape)

            print(label.shape)
        else:
            break
