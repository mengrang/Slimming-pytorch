# coding:utf-8
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
import os
import os.path as osp
from real_prune import *

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

def BatchNorm2d_no_b(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
    bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
    bn.bias.requires_grad = False
    return bn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, index, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, index[0], stride)
        self.bn1 = BatchNorm2d_no_b(index[0])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(index[1], planes)
        self.bn2 = BatchNorm2d_no_b(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, block_idx, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, block_idx[0], kernel_size=1, bias=True)
        self.bn1 = BatchNorm2d_no_b(idx[0])
        self.conv2 = nn.Conv2d(idx[0], block_idx[1], kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = BatchNorm2d_no_b(block_idx[1])
        self.conv3 = nn.Conv2d(block_idx[1], planes * 4, kernel_size=1, bias=True)
        self.bn3 = BatchNorm2d_no_b(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, net_index=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if net_index = None:
            ## Standard Resnet Structure
            net_index = [
                    [[64]], 
                    [[[64, 64]]*layers[0]], 
                    [[[128, 128]]*layers[1]], 
                    [[[256, 256]]*layers[2]], 
                    [[[512, 512]]*layers[3]]
                    ]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = BatchNorm2d_no_b(net_index[0][0][0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], net_index[1])
        self.layer2 = self._make_layer(block, 128, layers[1], net_index[2], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], net_index[3], stride=2)
        # modify the layer 4 stride 
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, layer_idx, stride=1):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, layer_idx[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, layer_idx[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   

        x = self.layer1(x)
        fm1 = x
        x = self.layer2(x)
        fm2 = x
        x = self.layer3(x)
        fm3 = x
        x = self.layer4(x)

        return x, (fm1, fm2, fm3)

def slim_finetune(model, ckpt_dir, ratio):
    model_dict = model.state_dict() 
    ckpt = torch.load(osp.join(ckpt_dir, "ckpt.pth"))
    ckpt_dict = ckpt['state_dicts'][0]
    bn_w_dict = bn_weights(ckpt_dict)
    bnw_index, slim_bnw_dict = slim_bnws(bn_w_dict, ratio)
    model_dict.update(slim_bnw_dict)
    return model_dict
    # bn_b_dict = dict()
    # for k, v in model_dict.items():
    #     for i in k.split('.'):
    #         if (i == 'bn1' or i == 'bn2' or i == 'bn3') and k.endswith('bias'):
    #             bn_b_dict[k] = v
    # pre_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in bn_b_dict}
    # model_dict.update(pre_dict)
    # return model_dict
    
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model.load_state_dict(slim_finetune(model, ckpt_dir, ratio))
    if pretrained:
        model.load_state_dict(removed_dict(model_urls['resnet50'], model))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model