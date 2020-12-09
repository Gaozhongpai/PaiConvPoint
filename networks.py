import torch.nn as nn
import torch 
import torch.nn.init as init
import torch.nn.functional as F
from util import knn

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
        bsize = x.size(0)

        x = self.conv1(x)                       # (bsize, 3*2, num_points, k) -> (bsize, 64, num_points, k)
        x = self.conv2(x)                       # (bsize, 64, num_points, k) -> (bsize, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (bsize, 128, num_points, k) -> (bsize, 128, num_points)

        x = self.conv3(x)                       # (bsize, 128, num_points) -> (bsize, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (bsize, 1024, num_points) -> (bsize, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (bsize, 1024) -> (bsize, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (bsize, 512) -> (bsize, 256)

        x = self.transform(x)                   # (bsize, 256) -> (bsize, 3*3)
        x = self.compute_rotation_matrix_from_ortho6d(x)
        return x


class TemperatureNet(nn.Module):
    def __init__(self, args):
        super(TemperatureNet, self).__init__()
        self.temp_factor = args.temp_factor
        self.nn = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1),
                                nn.BatchNorm1d(64),
                                nn.LeakyReLU(negative_slope=0.2))
        self.linear = nn.Linear(64, 1)
                                
    def forward(self, input):
        bsize = input.shape[0]
        x = input.permute(0, 2, 1).contiguous()
        x = self.nn(x)
        x = F.adaptive_max_pool1d(x, 1).view(bsize, -1)
        x = torch.sigmoid(self.linear(x)[:, :, None])*(0.1 - 1.0/self.temp_factor) + 1.0/self.temp_factor
        return x

def get_graph_feature(x, k=20, idx=None):
    bsize = x.size(0)
    num_points = x.size(2)
    x = x.view(bsize, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (bsize, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, bsize, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (bsize, num_points, num_dims)  -> (bsize*num_points, num_dims) #   bsize * num_points * k + range(0, bsize*num_points)
    feature = x.view(bsize*num_points, -1)[idx, :]
    feature = feature.view(bsize, num_points, k, num_dims) 
    x = x.view(bsize, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature