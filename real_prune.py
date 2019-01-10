# coding: utf-8
import numpy as np
import os
import torch.nn as nn
import torch
import torch.nn.functional as f
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from config import *
from distill import teacher, student
from dataset import dataset
from utils import *
import collections

def mask(var, thr):
    return list(var.cpu().abs().gt(thr).float().numpy())

def bn_channels(var, thr):
    var = mask(var, thr)
    return sum(var)

def bn_weights(state_dict):
    bn_w_dict = OrderedDict()
    for k, v in state_dict.items():
        for i in k.split('.'):
            if (i == 'bn1' or i == 'bn2') and k.endswith('weight'):
                bn_w_dict[k] = v
    return bn_w_dict

def slim_channels(bn_w_dict, layers, ratio, model='ResNet50'):
    block_clust = []
    layer_clust = []
    bnw_state = bn_state(bn_w_dict, ratio)
    if model == 'ResNet50':
        net_channels = [
                [[64]], 
                [[64, 64]]*layers[0], 
                [[128, 128]]*layers[1], 
                [[256, 256]]*layers[2], 
                [[512, 512]]*layers[3]]           
    flat_channels = [int(bn_channels(v, min(float(bnw_state[k][-1]), 0.05))) for k, v in bn_w_dict.items()]
    net_channels[0][0] = [flat_channels[0]]
    for i in range(1, len(flat_channels), 2):
        block_clust.append(flat_channels[i:i+2])
    layer_clust = [block_clust[:layers[0]]]
    for i in range(1, len(layers)): 
        layer_clust.append(block_clust[sum(layers[:i]):sum(layers[:i+1])])
    net_channels[1:] = layer_clust
    return net_channels

def slim_bnws(bn_w_dict, ratio):
    bnw_index = OrderedDict()
    bnw_state = bn_state(bn_w_dict, ratio)    
    bnw_index = {k:np.argwhere(v.abs().cpu().numpy() > min(float(bnw_state[k][-1]), 0.05))
                for k,v in bn_w_dict.items()}
    pruned_bn_w_dict = {k:v[bnw_index[k]] for k,v in bn_w_dict.items()}
    return bnw_index, pruned_bn_w_dict
    
def threshold_adap(arr, ratio):
    arr.sort()
    num = arr.size
    elm = arr[int(num*(1-ratio))]
    return elm

def slim_statistic(bn_w_dict, layers, ratio, model='ResNet50'):
    C_slim_ratio = []
    bnw_state = bn_state(bn_w_dict, ratio)
    if model == 'ResNet50':
        flat_net_C = [c for layer in ResNet50_C for block in layer for c in block]
    flat_slim_C = [bn_channels(v, float(bnw_state[k][-1])) for k, v in bn_w_dict.items()]
    for p in zip(flat_slim_C, flat_net_C):
        C_slim_ratio.append(round(p[0]/p[1], 4))
    total_C_slim_ratio = sum(C_slim_ratio) / len(flat_net_C)

    return C_slim_ratio, total_C_slim_ratio

def bn_state(bn_w_dict, ratio):
    bnw_state = OrderedDict()

    bnw_state = {k:[v.abs().cpu().numpy().max(), 
                v.abs().cpu().numpy().min(), 
                v.abs().cpu().numpy().mean(), 
                np.median(v.abs().cpu().numpy()),
                threshold_adap(v.abs().cpu().numpy(), ratio)] for k,v in bn_w_dict.items()}
    return bnw_state


if __name__ == "__main__": 
    ResNet50_LAYERS = [3, 4, 6, 3]
    ResNet50_C = [
            [[64]], 
            [[64, 64]]*ResNet50_LAYERS[0], 
            [[128, 128]]*ResNet50_LAYERS[1], 
            [[256, 256]]*ResNet50_LAYERS[2], 
            [[512, 512]]*ResNet50_LAYERS[3]
            ]
    ckpt = torch.load(osp.join("/home/zhangming/log/mr/train_slim/slim_baseline_v4", "ckpt.pth"))
    ckpt_dict = ckpt['state_dicts'][0]
 
    bn_w_dict = bn_weights(ckpt_dict)

    bnw_index, slim_bnw_dict = slim_bnws(bn_w_dict, 0.8)  
    net_channels = slim_channels(bn_w_dict, ResNet50_LAYERS, 0.8, model='ResNet50')
    print([
        [[64]], 
        [[64, 64]]*3, 
        [[128, 128]]*4, 
        [[256, 256]]*6, 
        [[512, 512]]*3])
    print(net_channels)
    C_slim_ratio, total_C_slim_ratio = slim_statistic(bn_w_dict, ResNet50_LAYERS, 0.8, model='ResNet50')
    print('Channels Slimming Ratio:\n{}'.format(C_slim_ratio))
    print('Total Channels Slimming Ratio:\n{:4f}'.format(total_C_slim_ratio))
    bnw_state = bn_state(bn_w_dict, 0.8)
    for k,v in bnw_state.items():
        print(k)
        print(v)
