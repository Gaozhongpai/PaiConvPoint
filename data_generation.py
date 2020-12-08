from tqdm import tqdm
import os, argparse
#from psbody.mesh import Mesh
import torch
import shutil
import glob
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import numpy as np


def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)

root = "/home/yyy/code/modelnet40_normal_resampled/Processed/sliced"
subsampling_parameter = 0.0046
datasets = os.listdir(os.path.join(root))
datasets.sort()
mini = 10000
i = 0

datasets = os.listdir(os.path.join(root))
datasets.sort()
for dataset in datasets:
    testers = os.listdir(os.path.join(root, dataset))
    testers.sort()
    for tester in testers:
        meshes = os.listdir(os.path.join(root, dataset, tester))
        meshes.sort()
        for mesh in meshes:
            data = np.loadtxt(os.path.join(root, dataset, tester, mesh), delimiter=',', dtype=np.float32)
            data = torch.from_numpy(data)
            torch.save(data, os.path.join(root, dataset, tester, mesh[:-3]+"tch"))

for dataset in datasets:
    testers = os.listdir(os.path.join(root, dataset))
    testers.sort()
    for tester in testers:
        meshes = glob.glob(os.path.join(root, dataset, tester, "*.tch"))
        meshes.sort()
        for mesh in meshes:
            data = torch.load(mesh)[:, :3]
            data = data - (torch.max(data, 0)[0] + torch.min(data, 0)[0]) / 2.
            data = 2. * data / (torch.max(data) - torch.min(data))
            data, _ = grid_subsampling(data[:, :3],
                                            features=data[:, 3:],
                                            sampleDl=subsampling_parameter)
            mini = data.shape[0] if mini > data.shape[0] else mini
            data = torch.from_numpy(data)
            torch.save(data, mesh[:-3]+"spl")
            i = i + 1
            if i % 100 == 0:
                print("we are at {} and mini is {}".format(i, mini))

testlist = np.loadtxt("/home/yyy/code/modelnet40_normal_resampled/Processed/modelnet40_test.txt", dtype='str')
testlist = list(testlist)
testers = os.listdir(os.path.join(root, "train"))
testers.sort()
if not os.path.exists(os.path.join(root, "test")):
    os.mkdir(os.path.join(root, "test"))
for tester in testers:
    meshes = os.listdir(os.path.join(root, "train", tester))
    meshes.sort()
    if not os.path.exists(os.path.join(root, "test", tester)):
        os.mkdir(os.path.join(root, "test", tester))
    for mesh in meshes:
        if mesh[:-4] in testlist:
            shutil.move(os.path.join(root, "train", tester, mesh), 
                os.path.join(root, "test", tester, mesh))
