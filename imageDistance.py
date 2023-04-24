import torch
import numpy as np
from collections import defaultdict
import time
import matplotlib.pyplot as plt

from utils import *

def image_distance_dataset():
    label_flag = defaultdict(int)
    label_dist = defaultdict(float)

    for i, data in enumerate(distributionloader, 0):
        input, label = data
        label = label.item()

        if label_flag[label]==1:
            continue
        else:
            label_flag[label] = 1
        
        print('distance for label%d at'%(label) , time.strftime('%Y-%m-%d %H:%M:%S'))

        candi_data = []
        candi_norm = []

        for i_origin, data_origin in enumerate(cifar100loader, 0):
            input_origin, label_origin = data_origin
            label_origin = label_origin.item()
            
            candi_data.append(data_origin)
            candi_norm.append(l2_norm(input_origin, input).item())
        
        min_idx = torch.argmin(torch.tensor(candi_norm))
        min_input, min_label = candi_data[min_idx.item()]

        imshow_2(torchvision.utils.make_grid(min_input), torchvision.utils.make_grid(input), 
                'label%d_minlabel%d'%(label, min_label), candi_norm[min_idx.item()])
        
        label_dist[label] = candi_norm[min_idx.item()]

        if sum(label_flag.values())==100: break

    print('randomly selected distribution-mathcing image per class(0-99) <--> most similar image in original set')
    print('min label:', torch.argmin(label_dist.values()))
    print('max label:', torch.argmin(label_dist.values()))

def image_distance(imgList):
    label_flag = defaultdict(int)
    label_dist = defaultdict(float)

    for i, data in enumerate(imgList, 0):
        input = data
        
        print('distance for %d image at %s'%(i , time.strftime('%Y-%m-%d %H:%M:%S')))

        candi_data = []
        candi_norm = []

        for i_origin, data_origin in enumerate(cifar100loader, 0):
            input_origin, label_origin = data_origin
            label_origin = label_origin.item()
            
            candi_data.append(data_origin)
            candi_norm.append(l2_norm(input_origin, input).item())
        
        min_idx = torch.argmin(torch.tensor(candi_norm))
        min_input, min_label = candi_data[min_idx.item()]

        imshow_2(torchvision.utils.make_grid(min_input), torchvision.utils.make_grid(input), 
                'distribution-generated/%dminlabel%d'%(i, min_label), candi_norm[min_idx.item()])
        
        label_dist[i] = candi_norm[min_idx.item()]

        if sum(label_flag.values())==100: break

    print('randomly selected distribution-mathcing image per class(0-99) <--> most similar image in original set')
    print('min label:', torch.argmin( torch.tensor(label_dist.values())) )
    print('max label:', torch.argmin( torch.tensor(label_dist.values())) )

