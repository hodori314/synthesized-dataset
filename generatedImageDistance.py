import torch
from utils import *
from imageDistance import *
from DCGANmodel import *

myImageList = []
b_size = 1
nz = 100

myNetG = Generator(0)
myNetG.load_state_dict(torch.load('netG.pth'))
myNetG.eval()

for i in range(100):
    rand_noise = torch.randn(b_size, nz, 1, 1)
    fake = myNetG(rand_noise)
    myImageList.append(fake)

image_distance(myImageList)