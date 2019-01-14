# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import resnet_bn_slim
from config import *

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.pretrained_model = resnet_bn_slim.resnet50(pretrained=False)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, 200)

    def forward(self, x):
        logits = self.pretrained_model(x)    
        return logits