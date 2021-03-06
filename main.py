#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
# from pai_model import PaiNet
from model_dilated import PaiNet
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from tqdm import tqdm
from networks import DGCNN


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    else:
        model = PaiNet(args).to(device)
        #raise Exception("Not implemented")
    print(str(model))
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params))
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # if os.path.exists('checkpoints/%s/models/model_%s.t7'% (args.exp_name, args.model)):
    #     checkpoint_dict = torch.load('./checkpoints/%s/models/model_%s.t7'% (args.exp_name, args.model), map_location=device)
    #     model_dict = model.state_dict()
    #     pretrained_dict = checkpoint_dict
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "one_padding" not in k}
    #     model_dict.update(pretrained_dict) 
    #     model.load_state_dict(pretrained_dict, strict=False)
    #     #model.load_state_dict(checkpoint_dict, strict=True)
    #     print("Load model from './checkpoints/%s/models/model_%s.t7 !'"% (args.exp_name, args.model))

    if args.use_sgd:
        print("Use SGD")

        trainables_wo_bn = [param for name, param in model.named_parameters()
                       if param.requires_grad \
                       and "bn" not in name \
                       and "kernals" not in name]
        trainables_wt_bn = [param for name, param in model.named_parameters()
                        if param.requires_grad and ('bn' in name or "kernals" in name)]
        opt = optim.SGD([{'params': trainables_wo_bn, 'weight_decay': 5e-5},
                          {'params': trainables_wt_bn}],
                          lr=args.lr*100, momentum=args.momentum)
        #opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = cal_loss

    best_test_acc = 0 # 0.931929
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for b, (data, label) in enumerate(tqdm(train_loader, ncols=0)):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        
        scheduler.step()
        # if epoch % 100 == 0:
        #     scheduler = CosineAnnealingLR(opt, 50, eta_min=args.lr)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, lr: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred), 
                                                                                scheduler.get_lr()[0])
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        with torch.no_grad():
            for data, label in test_loader:
        
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                logits = model(data)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model_%s.t7' % (args.exp_name, args.model))
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test best: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc, 
                                                                              best_test_acc)
        io.cprint(outstr)


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    else:
        model = PaiNet(args).to(device)
        #raise Exception("Not implemented")
    print(str(model))
    # model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    if os.path.exists('checkpoints/%s/models/model_%s.t7'% (args.exp_name, args.model)):
        checkpoint_dict = torch.load('./checkpoints/%s/models/model_%s.t7'% (args.exp_name, args.model), map_location=device)
        # pretrained_dict = {}
        # for k, v in checkpoint_dict.items():
        #     if 'transform' in k:
        #         k = k.replace('transform', 'paiIdxMatrix')
        #     pretrained_dict[k]=v           
        # # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'transform' in k}
        model.load_state_dict(checkpoint_dict, strict=True)
        # torch.save(model.state_dict(), 'checkpoints/%s/models/model_%s_2048.t7' % (args.exp_name, args.model))
        print("Load model from './checkpoints/%s/models/model_%s_2048.t7 !'"% (args.exp_name, args.model))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='dialted-1024-2048-16', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='paigcnn', metavar='N',
                        choices=['pointnet', 'paigcnn', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=24, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=24, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=2048, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=12, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--temp_factor', type=int, default=100, metavar='N',
                        help='Factor to control the softmax precision')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
