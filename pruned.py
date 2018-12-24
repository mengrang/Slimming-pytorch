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

def bn_weight(state_dict):
    bn_w_dict = dict()
    for k, v in state_dict.items():
        for i in k.split('.'):
            if (i == 'bn1' or i == 'bn2') and k.endswith('weight'):
                bn_w_dict[k] = v
    return bn_w_dict

def pruned_index(bn_w_dict):
    pruned_dict = dict()
    index = []
    for k, v in bn_w_dict.items():
        mask = v.abs().gt(THRESHOLD).float().cuda()
        pruned_dict[k] = mask
    return pruned_dict

# def real_pruning(ckpt_dict, pruned_dict):   
#     for k, v in ckpt_dict.items():
#         if k in pruned_dict.keys():

if __name__ == "__main__":
    ckpt = torch.load('E:\save\distill_slim\\t_s_slim_save_models_resnet50\\20181221_205447\\021.ckpt')
    ckpt_dict = ckpt['state_dict']
    bn_w_dict = bn_weight(ckpt_dict)     
    pruned_dict = pruned_index(bn_w_dict)
    print(pruned_dict)   

