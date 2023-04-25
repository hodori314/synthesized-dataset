import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.models as models

import argparse
import os
import time

import numpy as np
import pandas as pd
# from tsne import bh_sne
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch t-SNE for STL10')
parser.add_argument('--save-dir', type=str, default='./results', help='path to save the t-sne image')
parser.add_argument('--batch-size', type=int, default=32, help='batch size (default: 128)')
parser.add_argument('--seed', type=int, default=1, help='random seed value (default: 1)')
parser.add_argument('--savefile', type=int, default=0, help='is there any save file?(default: 0)')
parser.add_argument('--step', type=int, default=0, help='using save file in the gen function(default: 0)')

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

device = 'cpu'

# set seed
torch.manual_seed(args.seed)
if device == 'cuda':
    torch.cuda.manual_seed(args.seed)

# set dataset

batch_size = 1
image_size = 32
num_workers = 0

transform = transforms.Compose(
    [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar100set = torchvision.datasets.CIFAR100(root='cifar100', train=True, download=False, transform=transform)
cifar100loader = torch.utils.data.DataLoader(cifar100set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

distributionset = dset.ImageFolder(root='distribution-matching', transform=transform)
distributionloader =  torch.utils.data.DataLoader(distributionset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

generatedset = dset.ImageFolder(root='distribution-generated', transform=transform)
generatedloader = torch.utils.data.DataLoader(generatedset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

cGeneratedset = dset.ImageFolder(root='cond-gen-samples', transform=transform)
cGeneratedloader = torch.utils.data.DataLoader(cGeneratedset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# set model
net = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet32", pretrained=True)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True



######################### TSNE #########################

textLabels = ['apple', 'bicycle', 'leopard']
distributionMatchingLabels = [0, 8, 42]
cifarLabels = [0, 78, 37]

def gen_features():
    net.eval()
    targets_list = []
    outputs_list = []

    if args.step <= 1:
        ## STEP 1. distribution data
        print('>>> STEP 1')
        dataloader = distributionloader
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                targets_np = targets.data.cpu().numpy()

                if targets_np[0] not in distributionMatchingLabels: continue
                else: print(targets_np, ' in ', distributionMatchingLabels)

                outputs = net(inputs)
                outputs_np = outputs.data.cpu().numpy()
                
                targets_list.append(targets_np[:, np.newaxis])
                outputs_list.append(outputs_np)
                
                if ((idx+1) % 10 == 0) or (idx+1 == len(dataloader)):
                    print(idx+1, '/', len(dataloader), time.strftime('%Y-%m-%d %H:%M:%S'))

                # ### FOR DEBUG
                # if idx > 10:
                #     break
        np.save('targets_1', np.array(targets_list))
        np.save('outputs_1', np.array(outputs_list))

    if args.step <=2 :
        ## STEP 2. cifar100 
        print('>>> STEP 2')

        dataloader = cifar100loader
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                targets_np = targets.data.cpu().numpy()

                if targets_np[0] not in cifarLabels: continue
                else: print(targets_np, ' in ', cifarLabels)

                if targets_np[0]==0:
                    for i in range(len(targets_np)):
                        targets_np[i] += 1

                outputs = net(inputs)
                outputs_np = outputs.data.cpu().numpy()
                
                targets_list.append(targets_np[:, np.newaxis])
                outputs_list.append(outputs_np)
                
                if ((idx+1) % 10 == 0) or (idx+1 == len(dataloader)):
                    print(idx+1, '/', len(dataloader), time.strftime('%Y-%m-%d %H:%M:%S'))

                # ### FOR DEBUG
                # if idx > 10:
                #     break

        np.save('targets_2', np.array(targets_list))
        np.save('outputs_2', np.array(outputs_list))

    if args.step <= 3:
        targets_list = list( np.load('tsne-save/targets_2.npy'))
        outputs_list = list( np.load('tsne-save/outputs_2.npy'))
        ## STEP 3. generated set
        print('>>> STEP 3')

        dataloader = cGeneratedloader
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                targets_np = targets.data.cpu().numpy()
                for i in range(len(targets_np)):
                    targets_np[i] += 100
                print(targets_np)

                # print(idx)

                outputs = net(inputs)
                outputs_np = outputs.data.cpu().numpy()
                
                targets_list.append(targets_np[:, np.newaxis])
                outputs_list.append(outputs_np)
                
                if ((idx+1) % 10 == 0) or (idx+1 == len(dataloader)):
                    print(idx+1, '/', len(dataloader), time.strftime('%Y-%m-%d %H:%M:%S'))
        
        np.save('targets_3', np.array(targets_list))
        np.save('outputs_3', np.array(outputs_list))

    

    targets = np.concatenate(targets_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)

    np.save('targets', targets)
    np.save('outputs', outputs)

    return targets, outputs

def tsne_plot(save_dir, targets, outputs):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    tsne = TSNE(random_state=0, perplexity=5)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette("hls", 7),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(os.path.join(save_dir,'tsne-cifar100-dm-cg.png'), bbox_inches='tight')
    print('done!')

if args.savefile:
    targets = np.load('targets.npy')
    outputs = np.load('outputs.npy')
else:
    targets, outputs = gen_features()


tsne_plot(args.save_dir, targets, outputs)