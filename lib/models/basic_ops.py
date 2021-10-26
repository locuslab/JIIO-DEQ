import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
# import ipdb
from optimizations import *

BN_MOMENTUM = 0.1
DEQ_EXPAND = 5        # Don't change the value here. The value is controlled by the yaml files.
NUM_GROUPS = 4        # Don't change the value here. The value is controlled by the yaml files.
ENCODING_CHANNELS = 400

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)

def convTranspose3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convTranspose with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias, output_padding=stride // 2)

def norm_diff(new, old, show_list=False):
    if show_list:
        return [(new[i] - old[i]).norm().item() for i in range(len(new))]
    return np.sqrt(sum((new[i] - old[i]).norm().item()**2 for i in range(len(new))))

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, gn=False):
        """
        A bottleneck block with receptive field only 3x3. (This is not used in MDEQ; only
        in the classifier layer).
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if gn:
            self.gn1 = nn.GroupNorm(NUM_GROUPS, planes, affine=False)
            self.norm1 = self.gn1
        else:
            self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=False)
            self.norm1 = self.bn1

        self.conv2 = conv3x3(planes, planes, stride=stride)
        if gn:
            self.gn2 = nn.GroupNorm(NUM_GROUPS, planes, affine=False)
            self.norm2 = self.gn2
        else:
            self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=False)
            self.norm2 = self.bn2

        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        if gn:
            self.gn3 = nn.GroupNorm(NUM_GROUPS, planes*self.expansion, affine=False)
            self.norm3 = self.gn3
        else:
            self.bn3 = nn.BatchNorm2d(planes*self.expansion, momentum=BN_MOMENTUM, affine=False)
            self.norm3 = self.bn3
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, injection=None):
        if injection is None:
            injection = 0
        residual = x

        out = self.conv1(x) + injection
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout=0.0, wnorm=False,num_branches=1):
        """
        A canonical residual block with two 3x3 convolutions and an intermediate ReLU. 
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, DEQ_EXPAND*planes, stride)
        self.gn1 = nn.GroupNorm(NUM_GROUPS, DEQ_EXPAND*planes, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(DEQ_EXPAND*planes, planes)
        self.gn2 = nn.GroupNorm(NUM_GROUPS, planes, affine=True)
        
        self.downsample = downsample
        self.stride = stride
        
        self.gn3 = nn.GroupNorm(NUM_GROUPS, planes, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop = VariationalHidDropout2d(dropout)
        if wnorm: self._wnorm()
    
    def _wnorm(self):
        self.conv1, self.conv1_fn = weight_norm(self.conv1, names=['weight'], dim=0)
        self.conv2, self.conv2_fn = weight_norm(self.conv2, names=['weight'], dim=0)
    
    def _reset(self, x):
        if 'conv1_fn' in self.__dict__:
            self.conv1_fn.reset(self.conv1)
        if 'conv2_fn' in self.__dict__:
            self.conv2_fn.reset(self.conv2)
        self.drop.reset_mask(x)
    
    def _copy(self, other):
        self.conv1.weight.data = other.conv1.weight.data.clone()
        self.conv2.weight.data = other.conv2.weight.data.clone()
        self.drop.mask = other.drop.mask.clone()
        if self.downsample:
            assert False, "Shouldn't be here. Check again"
            self.downsample.weight.data = other.downsample.weight.data
        for i in range(1,4):
            try:
                eval(f'self.gn{i}').weight.data = eval(f'other.gn{i}').weight.data.clone()
                eval(f'self.gn{i}').bias.data = eval(f'other.gn{i}').bias.data.clone()
            except:
                print(f"Did not set affine=True for gnorm(s) in gn{i}?")
            
    def forward(self, x, injection=None, *args):
        if injection is None:
            injection = 0
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        
        out = self.drop(self.conv2(out)) + injection
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gn3(self.relu3(out))

        return out


class BasicMetaBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout=0.0, wnorm=False, num_branches=1):
        """
        A canonical residual block with two 3x3 convolutions and an intermediate ReLU and Group Normalizations, 
        with a FILM layer after the first group norm. The context vector in the film layer is used as the task 
        representation which is optimized in the inner loop. """
        super(BasicMetaBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, DEQ_EXPAND*planes, stride)
        self.gn1 = nn.GroupNorm(NUM_GROUPS, DEQ_EXPAND*planes, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.film1 = nn.Linear(ENCODING_CHANNELS//(num_branches), DEQ_EXPAND*planes*2)
        
        self.conv2 = conv3x3(DEQ_EXPAND*planes, planes)
        self.gn2 = nn.GroupNorm(NUM_GROUPS, planes, affine=True)
        self.film2 = nn.Linear(ENCODING_CHANNELS//(num_branches), planes*2)
        
        self.downsample = downsample
        self.stride = stride
        
        self.gn3 = nn.GroupNorm(NUM_GROUPS, planes, affine=True)
        self.film3 = nn.Linear(ENCODING_CHANNELS//num_branches, planes*2)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop = VariationalHidDropout2d(dropout)
        if wnorm: self._wnorm()
    
    def _wnorm(self):
        self.conv1, self.conv1_fn = weight_norm(self.conv1, names=['weight'], dim=0)
        self.conv2, self.conv2_fn = weight_norm(self.conv2, names=['weight'], dim=0)
    
    def _reset(self, x):
        if 'conv1_fn' in self.__dict__:
            self.conv1_fn.reset(self.conv1)
        if 'conv2_fn' in self.__dict__:
            self.conv2_fn.reset(self.conv2)
        self.drop.reset_mask(x)
    
    def _copy(self, other):
        self.conv1.weight.data = other.conv1.weight.data.clone()
        self.conv2.weight.data = other.conv2.weight.data.clone()
        self.drop.mask = other.drop.mask.clone()
        if self.downsample:
            assert False, "Shouldn't be here. Check again"
            self.downsample.weight.data = other.downsample.weight.data
        for i in range(1,4):
            try:
                eval(f'self.gn{i}').weight.data = eval(f'other.gn{i}').weight.data.clone()
                eval(f'self.gn{i}').bias.data = eval(f'other.gn{i}').bias.data.clone()
            except:
                print(f"Did not set affine=True for gnorm(s) in gn{i}?")
        self.film1.weight.data = other.film1.weight.data.clone()  
        self.film2.weight.data = other.film2.weight.data.clone()
        self.film3.weight.data = other.film3.weight.data.clone()
        self.film1.bias.data = other.film1.bias.data.clone()  
        self.film2.bias.data = other.film2.bias.data.clone()
        self.film3.bias.data = other.film3.bias.data.clone()
            
    def forward(self, x, injection=None, context=None):
        if injection is None:
            injection = 0
        residual = x
        if not context.shape[0]==x.shape[0]:
            context = context.repeat_interleave(x.shape[0]//context.shape[0], 0)
        
        out = self.conv1(x)
        out = self.gn1(out)
        film1 = (self.film1(context)).view(x.shape[0], -1, 1, 1)
        gamma1, beta1 = torch.chunk(film1, 2, dim=1)
        out = out*gamma1 + beta1
        out = self.relu(out)
        
        out = self.drop(self.conv2(out)) + injection
        out = self.gn2(out)

        # Tried adding film2 and film3, but didn't really observe any boost in performance.
        # However, note that when using film2/film3, need to use a separate context vector for this.
        # Can't share context vectors between different film layers (results in instability)
        
        # film2 = self.film2(context2).view(x.shape[0], -1, 1, 1)
        # gamma2, beta2 = torch.chunk(film2, 2, dim=1)
        # out = out*gamma2 + beta2

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gn3(self.relu3(out))

        # film3 = self.film3(context).view(x.shape[0], -1, 1, 1)
        # gamma3, beta3 = torch.chunk(film3, 2, dim=1)
        # out = out*gamma3 + beta3

        return out
