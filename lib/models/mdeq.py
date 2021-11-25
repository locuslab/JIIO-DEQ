from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import functools
from termcolor import colored

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
sys.path.append("lib/models")
from mdeq_core import MDEQNet, MDEQJIIOVAECore, MDEQIMAMLCore
from basic_ops import *

logger = logging.getLogger(__name__)

class MDEQClsNet(MDEQNet):
    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ Classification model with the given hyperparameters
        """
        global BN_MOMENTUM
        super(MDEQClsNet, self).__init__(cfg, BN_MOMENTUM=BN_MOMENTUM, **kwargs)
        self.head_channels = cfg['MODEL']['EXTRA']['FULL_STAGE']['HEAD_CHANNELS']
        self.final_chansize = cfg['MODEL']['EXTRA']['FULL_STAGE']['FINAL_CHANSIZE']

        # Classification Head
        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(self.num_channels)
        self.classifier = nn.Linear(self.final_chansize, self.num_classes)
            
    def _make_head(self, pre_stage_channels):
        """
        Create a classification head that:
           - Increase the number of features in each resolution 
           - Downsample higher-resolution equilibria to the lowest-resolution and concatenate
           - Pass through a final FC layer for classification
        """
        head_block = Bottleneck
        d_model = self.init_chansize
        head_channels = self.head_channels
        
        # Increasing the number of channels on each resolution when doing classification. 
        incre_modules = []
        for i, channels  in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block, channels, head_channels[i], blocks=1, stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
            
        # Downsample the high-resolution streams to perform classification
        downsamp_modules = []
        for i in range(len(pre_stage_channels)-1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i+1] * head_block.expansion

            downsamp_module = nn.Sequential(conv3x3(in_channels, out_channels, stride=2, bias=True),
                                            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                                            nn.ReLU(inplace=True))
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        # Final FC layers
        final_layer = nn.Sequential(nn.Conv2d(head_channels[len(pre_stage_channels)-1] * head_block.expansion,
                                              self.final_chansize,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0),
                                    nn.BatchNorm2d(self.final_chansize, momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True))
        return incre_modules, downsamp_modules, final_layer

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x, train_step=0, **kwargs):
        y_list = self._forward(x, train_step, **kwargs)
        
        # Classification Head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i+1](y_list[i+1]) + self.downsamp_modules[i](y)
        y = self.final_layer(y)

        # Pool to a 1x1 vector (if needed)
        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
        y = self.classifier(y)
        
        return y
    
    def init_weights(self, pretrained='',):
        """
        Model initialization. If pretrained weights are specified, we load the weights.
        """
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            

class MDEQSegNet(MDEQNet):
    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ Segmentation model with the given hyperparameters
        """
        global BN_MOMENTUM
        super(MDEQSegNet, self).__init__(cfg, BN_MOMENTUM=BN_MOMENTUM, **kwargs)
        extra = cfg.MODEL.EXTRA
        
        # Last layer
        last_inp_channels = np.int(np.sum(self.num_channels))
        self.last_layer = nn.Sequential(nn.Conv2d(last_inp_channels, last_inp_channels, kernel_size=1),
                                        nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(last_inp_channels, cfg.DATASET.NUM_CLASSES, extra.FINAL_CONV_KERNEL, 
                                                  stride=1, padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0))
               
    def forward(self, x, train_step=0, **kwargs):
        y = self._forward(x, train_step, **kwargs)
        
        # Segmentation Head
        y0_h, y0_w = y[0].size(2), y[0].size(3)
        all_res = [y[0]]
        for i in range(1, self.num_branches):
            all_res.append(F.interpolate(y[i], size=(y0_h, y0_w), mode='bilinear', align_corners=True))

        y = torch.cat(all_res, dim=1)
        all_res = None
        # torch.cuda.empty_cache()
        y = self.last_layer(y)
        return y
    
    def init_weights(self, pretrained=''):
        """
        Model initialization. If pretrained weights are specified, we load the weights.
        """
        logger.info(f'=> init weights from normal distribution. PRETRAINED={pretrained}')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            
            # Just verification...
            diff_modules = set()
            for k in pretrained_dict.keys():
                if k not in model_dict.keys():
                    diff_modules.add(k.split(".")[0])
            print(colored(f"In ImageNet MDEQ but not Cityscapes MDEQ: {sorted(list(diff_modules))}", "red"))
            diff_modules = set()
            for k in model_dict.keys():
                if k not in pretrained_dict.keys():
                    diff_modules.add(k.split(".")[0])
            print(colored(f"In Cityscapes MDEQ but not ImageNet MDEQ: {sorted(list(diff_modules))}", "green"))
            
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

class MDEQJIIOVAE(MDEQJIIOVAECore):
    def __init__(self, cfg, **kwargs):
        super(MDEQJIIOVAE, self).__init__(cfg, **kwargs)

    def forward(self, x=None, train_step=-1, latent=None, **kwargs):
        vae_fwd = kwargs.get('vae_fwd',False)
        if vae_fwd:
            return self.vae_forward(x, train_step, **kwargs)
        get_latent = kwargs.get('get_latent', False)
        out = self._deq_constr_opt(x=x, train_step=train_step, latent=latent, **kwargs)
        if get_latent:
            return self.enc_dec_deq.new_u
        return out

    def vae_forward(self, x, train_step=-1, **kwargs):
        mu, logsd, y = self._forward(x, train_step, **kwargs)
        kld_loss = self._calc_kld(mu, logsd)
        return kld_loss, y
    
    def constr_opt(self, x, latent=None, train_step=-1, **kwargs):
        y_optimized = self._deq_constr_opt(x, latent=latent, train_step=train_step, **kwargs)
        return y_optimized
    
    def sample(self, sample_size, device='cuda', **kwargs):
        sampled_latent = torch.randn(sample_size, self.encoding_channels).to(device)
        return self._decode(sampled_latent, -1, **kwargs)
    
    def _calc_kld(self, mu, logsd):
        return torch.sum(mu**2 + logsd.exp()**2 - 2 * logsd - 1, dim=-1) / 2.
    
    def init_weights(self, pretrained=''):
        """
        Model initialization. If pretrained weights are specified, we load the weights.
        """
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


class MDEQRobustClsNet(MDEQNet):
    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ Classification model with the given hyperparameters
        """
        global BN_MOMENTUM
        super(MDEQRobustClsNet, self).__init__(cfg, BN_MOMENTUM=BN_MOMENTUM, **kwargs)

    def forward(self, x, y=None, train_step=0, **kwargs):
        y, jl = self._forward(x, y, train_step, **kwargs)
        return y, jl
    
    def init_weights(self, pretrained='',):
        """
        Model initialization. If pretrained weights are specified, we load the weights.
        """
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            

class MDEQIMAML(MDEQIMAMLCore):
    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ Classification model with the given hyperparameters
        """
        super(MDEQIMAML, self).__init__(cfg, BN_MOMENTUM=BN_MOMENTUM, **kwargs)

    def forward(self, x_tr, y_tr, x_te, train_step=0, **kwargs):
        return self._forward(x_tr, y_tr, x_te, train_step, **kwargs)
        
        
    def init_weights(self, pretrained=''):
        """
        Model initialization. If pretrained weights are specified, we load the weights.
        """
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):# or isinstance(m, nn.Linear) :
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

def get_cls_net(config, **kwargs):
    global BN_MOMENTUM
    BN_MOMENTUM = 0.1
    model = MDEQClsNet(config, **kwargs)
    model.init_weights()
    return model


def get_robust_cls_net(config, **kwargs):
    global BN_MOMENTUM
    BN_MOMENTUM = 0.1
    model = MDEQRobustClsNet(config, **kwargs)
    model.init_weights()
    return model


def get_seg_net(config, **kwargs):
    global BN_MOMENTUM
    BN_MOMENTUM = 0.01
    model = MDEQSegNet(config, **kwargs)
    model.init_weights(config.MODEL.PRETRAINED)
    return model

def get_jiio_vae(config, **kwargs):
    global BN_MOMENTUM
    BN_MOMENTUM = 0.01
    model = MDEQJIIOVAE(config, **kwargs)
    model.init_weights()
    return model


def get_imaml(config, **kwargs):
    global BN_MOMENTUM
    BN_MOMENTUM = 0.1
    model = MDEQIMAML(config, **kwargs)
    model.init_weights()
    return model

