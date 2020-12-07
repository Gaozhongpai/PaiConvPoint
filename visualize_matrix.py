#%%
import seaborn as sns
import numpy as np
import torch
import math, random
import matplotlib.pyplot as plt
from sparsemax import Sparsemax

def fibonacci_sphere(samples=1,randomize=False):
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = [[0, 0, 0]]
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(samples-1):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])

    return points

def TopMax(adjweight):
    # adjweight = (adjweight - torch.min(adjweight, dim=1, keepdim=True)[0]) / \
    #             (torch.max(adjweight, dim=1, keepdim=True)[0] - torch.min(adjweight, dim=1, keepdim=True)[0])  
    adjweight = torch.where(adjweight > 0, adjweight, torch.full_like(adjweight, 0.))  # adjweight[adjweight < 0] = torch.min(adjweight)*5
    adjweight = adjweight / (torch.sum(adjweight, dim=1, keepdim=True) + 1e-6)
    adjweight = adjweight * adjweight
    adjweight = adjweight / (torch.sum(adjweight, dim=1, keepdim=True) + 1e-6)
    adjweight = torch.where(adjweight > 0.1, adjweight, torch.full_like(adjweight, 0.)) 
    return adjweight

#%%
kernel_size = 20
t = 1./100
one_padding = torch.zeros(kernel_size, kernel_size)
one_padding.data[0, 0] = 1

kernels = torch.tensor(fibonacci_sphere(kernel_size)).transpose(0, 1)
#%%
neighbors = torch.rand([kernel_size, 3]) - 0.5
adjweight = torch.matmul(neighbors, kernels)
adjweight = (adjweight + one_padding) #

# adjweight = Sparsemax(-1)(adjweight.unsqueeze(0)).squeeze()
#out = torch.softmax(inp/t, dim=1)
adjweight = torch.softmax(adjweight/t, dim=0)
adjweight = torch.where(adjweight > 0.1, adjweight, torch.full_like(adjweight, 0.))
adjweight = adjweight / (torch.sum(adjweight, dim=0, keepdim=True) + 1e-6) 
# adjweight = Sparsemax(-1)(adjweight/t)
# adjweight = TopMax(adjweight.unsqueeze(0)).squeeze()
sns.heatmap(adjweight.numpy())
# print(adjweight)



# %%
