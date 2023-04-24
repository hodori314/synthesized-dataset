import torch
import torch.nn as nn
import torchvision

import torchvision.transforms as transforms
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import numpy as np

batch_size = 4
image_size = 64

transform = transforms.Compose(
    [
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar100set = torchvision.datasets.CIFAR100(root='./cifar100', train=True, download=False, transform=transform)
cifar100loader = torch.utils.data.DataLoader(cifar100set, batch_size=batch_size, shuffle=False, num_workers=0)

distributionset = dset.ImageFolder(root='distribution-matching', transform=transform)
distributionloader =  torch.utils.data.DataLoader(distributionset, batch_size=batch_size, shuffle=False, num_workers=0)


def l2_norm(x, y):
    v = x - y
    return torch.norm(v, p=2)

def imshow(img, file_name):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('%s.png'%(file_name))

def imshow_2(img1, img2, file_name, dist):
    img1 = img1 / 2 + 0.5
    npimg1 = img1.detach().numpy()
    img2 = img2 / 2 + 0.5
    npimg2 = img2.detach().numpy()

    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.title("original images")
    plt.imshow(np.transpose(npimg1, (1, 2, 0)))
    
    plt.subplot(1,2,2)
    plt.title("synthesized images with distance %.3f"%(dist))
    plt.imshow(np.transpose(npimg2, (1, 2, 0)))
    
    plt.savefig('%s.png'%(file_name))