#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:46:28 2018

@author: kg
"""
import argparse
import os
from utils2 import progress_bar

parser = argparse.ArgumentParser(description='PyTorch DepthGraph Training')
parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--trs', default=50, type=int, help='training batch size')
parser.add_argument('--tes', default=50, type=int, help='testing batch size')
parser.add_argument('--epoch', default=10, type=int, help='epoch times')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
not_save = 0
lr = args.lr

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from skimage import io

from models import FAN, ResNetDepth
import dataloader


use_cuda = torch.cuda.is_available()
best_loss = 100000  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print("Loading data...\n")
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor()
])
trainset = dataloader.MyDepthDataSet(train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.trs, shuffle=True)
testset = dataloader.MyDepthDataSet(train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.tes, shuffle=True)
print("Succeeded!\n")

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.dp')
#    net = checkpoint['net']
    net = FAN()
    state_dict = torch.load('./checkpoint/dict.dp')
    net.load_state_dict(state_dict)
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch'] + 1
else:
    print('==> Building model..')
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    net = FAN()


if use_cuda:
    net.cuda()
    print("Use CUDA:\n")
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
else:
    print("CUDA is disabled.\n")

criterion = nn.MSELoss(size_average=True, reduce=True)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def Train(epoch):
    print('\nEpoch: %d for Training' % epoch)
    net.train()
    train_loss = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        outputs = outputs[-1]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]

        progress_bar(batch_idx, len(trainloader), 'Avg Loss: %.5f| Loss: %.5f'
            % (train_loss / (batch_idx+1), loss.data[0]) )

#evaluating
def Test(epoch):
    print('\nEpoch: %d for Testing' % epoch)
    global best_loss, not_save, lr
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        outputs = outputs[-1]
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]

        progress_bar(batch_idx, len(testloader), 'Avg Loss: %.5f| Loss: %.5f'
            % (test_loss / (batch_idx+1), loss.data[0]) )

    avg_loss = test_loss / len(testloader)
    if(avg_loss < best_loss):
        print('Saving...')
        state = {
            'net': net.module if use_cuda else net,
            'loss': avg_loss,
            'epoch': epoch
        }
        state_dict = (net.module if use_cuda else net).state_dict()
        torch.save(state, './checkpoint/ckpt.dp')
        torch.save(state_dict, './checkpoint/dict.dp')
        best_loss = avg_loss
        not_save = 0
    else:
        not_save += 1
        if not_save > 15:
            lr /= 10
            optimizer.param_groups['lr'] = lr
            not_save = 0
    pass



for e in range(start_epoch, start_epoch+args.epoch):
    Train(e)
    Test(e)
