from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import copy
import logging
import functools

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from torchvision.utils import save_image
sys.path.append("lib/models")
sys.path.append("lib/modules")
sys.path.append("../modules")
from optimizations import *
from deq2d import *
from mdeq_forward_backward import MDEQWrapper, MDEQOptWrapper, MDEQClsWrapper, MDEQIMAMLWrapper
from basic_ops import *

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
       
blocks_dict = {
    'BASIC': BasicBlock,
    'MetaBASIC': BasicMetaBlock
}


class BranchNet(nn.Module):
    def __init__(self, blocks):
        """
        The residual block part of each resolution stream 
        (can optionally take a context vector v used in Meta Learning tasks)
        """
        super().__init__()
        self.blocks = blocks
    
    def forward(self, x, injection=None, v=None):
        blocks = self.blocks
        y = blocks[0](x, injection, v)
        for i in range(1, len(blocks)):
            y = blocks[i](y)
        return y

    
class DownsampleModule(nn.Module):
    def __init__(self, num_channels, in_res, out_res):
        """
        A downsample step from resolution j (with in_res) to resolution i (with out_res). A series of 2-strided convolutions.
        """
        super(DownsampleModule, self).__init__()
        # downsample (in_res=j, out_res=i)
        conv3x3s = []
        inp_chan = num_channels[in_res]
        out_chan = num_channels[out_res]
        self.level_diff = level_diff = out_res - in_res
        
        kwargs = {"kernel_size": 3, "stride": 2, "padding": 1, "bias": False}
        for k in range(level_diff):
            intermediate_out = out_chan if k == (level_diff-1) else inp_chan
            components = [('conv', nn.Conv2d(inp_chan, intermediate_out, **kwargs)), 
                          ('gnorm', nn.GroupNorm(NUM_GROUPS, intermediate_out, affine=True))]
            if k != (level_diff-1):
                components.append(('relu', nn.ReLU(inplace=True)))
            conv3x3s.append(nn.Sequential(OrderedDict(components)))
        self.net = nn.Sequential(*conv3x3s)  
        
    def _copy(self, other):
        for k in range(self.level_diff):
            self.net[k].conv.weight.data = other.net[k].conv.weight.data.clone()
            try:
                self.net[k].gnorm.weight.data = other.net[k].gnorm.weight.data.clone()
                self.net[k].gnorm.bias.data = other.net[k].gnorm.bias.data.clone()
            except:
                print("Did not set affine=True for gnorm(s)?")
            
    def forward(self, x):
        return self.net(x)


class UpsampleModule(nn.Module):
    def __init__(self, num_channels, in_res, out_res):
        """
        An upsample step from resolution j (with in_res) to resolution i (with out_res). 
        Simply a 1x1 convolution followed by an interpolation.
        """
        super(UpsampleModule, self).__init__()
        # upsample (in_res=j, out_res=i)
        inp_chan = num_channels[in_res]
        out_chan = num_channels[out_res]
        self.level_diff = level_diff = in_res - out_res
        
        self.net = nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(inp_chan, out_chan, kernel_size=1, bias=False)),
                        ('gnorm', nn.GroupNorm(NUM_GROUPS, out_chan, affine=True)),
                        ('upsample', nn.Upsample(scale_factor=2**level_diff, mode='nearest'))
                   ]))
    
    def _copy(self, other):
        self.net.conv.weight.data = other.net.conv.weight.data.clone()
        try:
            self.net.gnorm.weight.data = other.net.gnorm.weight.data.clone()
            self.net.gnorm.bias.data = other.net.gnorm.bias.data.clone()
        except:
            print("Did not set affine=True for gnorm(s)?")
        
    def forward(self, x):
        return self.net(x)

    
class MDEQModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_channels, fuse_method, dropout=0.0):
        """
        An MDEQ layer (note that MDEQ only has one layer). 
        """
        super(MDEQModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_channels)

        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.num_channels = num_channels

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels, dropout=dropout)
        self.fuse_layers = self._make_fuse_layers()
        self.post_fuse_layers = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('relu', nn.ReLU(False)),
                ('conv', nn.Conv2d(num_channels[i], num_channels[i], kernel_size=1, bias=False)),
                ('gnorm', nn.GroupNorm(NUM_GROUPS // 2, num_channels[i], affine=True))
            ])) for i in range(num_branches)])
        self.relu = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks, num_channels):
        """
        To check if the config file is consistent
        """
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _wnorm(self):
        """
        Apply weight normalization to the learnable parameters of MDEQ
        """
        self.post_fuse_fns = []
        for i, branch in enumerate(self.branches):
            for block in branch.blocks:
                block._wnorm()
            conv, fn = weight_norm(self.post_fuse_layers[i].conv, names=['weight'], dim=0)
            self.post_fuse_fns.append(fn)
            self.post_fuse_layers[i].conv = conv
        
        # Throw away garbage
        torch.cuda.empty_cache()
    
    def _copy(self, other):
        """
        Copy the parameter of an MDEQ layer. First copy the residual block, then the multiscale fusion part.
        """
        num_branches = self.num_branches
        for i, branch in enumerate(self.branches):
            for j, block in enumerate(branch.blocks):
                # Step 1: Basic block copying
                block._copy(other.branches[i].blocks[j])    
        
        for i in range(num_branches):
            for j in range(num_branches):
                # Step 2: Fuse layer copying
                if i != j:
                    self.fuse_layers[i][j]._copy(other.fuse_layers[i][j])     
            self.post_fuse_layers[i].conv.weight.data = other.post_fuse_layers[i].conv.weight.data.clone()
            try:
                self.post_fuse_layers[i].gnorm.weight.data = other.post_fuse_layers[i].gnorm.weight.data.clone()
                self.post_fuse_layers[i].gnorm.bias.data = other.post_fuse_layers[i].gnorm.bias.data.clone()
            except:
                print("Did not set affine=True for gnorm(s)?")
        
    def _reset(self, xs):
        """
        Reset the dropout mask and the learnable parameters (if weight normalization is applied)
        """
        for i, branch in enumerate(self.branches):
            for block in branch.blocks:
                block._reset(xs[i])
            if 'post_fuse_fns' in self.__dict__:
                self.post_fuse_fns[i].reset(self.post_fuse_layers[i].conv)    # Re-compute (...).conv.weight using _g and _v

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1, dropout=0.0, num_branches=1):
        layers = nn.ModuleList()
        n_channel = num_channels[branch_index]
        for i in range(num_blocks[branch_index]):
            layers.append(block(n_channel, n_channel, dropout=dropout, num_branches=num_branches))
        return BranchNet(layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels, dropout=0.0):
        """
        Make the residual block (s; default=1 block) of MDEQ's f_\theta layer
        """
        branch_layers = [self._make_one_branch(i, block, num_blocks, 
                                               num_channels, dropout=dropout, num_branches=num_branches) for i in range(num_branches)]
        
        # branch_layers[i] gives the module that operates on input from resolution i
        return nn.ModuleList(branch_layers)

    def _make_fuse_layers(self):
        """
        Create the multiscale fusion layer (which does simultaneous up- and downsamplings).
        """
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_channels = self.num_channels
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []                    # The fuse modules into branch #i
            for j in range(num_branches):
                if i == j:
                    fuse_layer.append(None)    # Identity if the same branch
                else:
                    module = UpsampleModule if j > i else DownsampleModule
                    fuse_layer.append(module(num_channels, in_res=j, out_res=i))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        # fuse_layers[i][j] gives the (series of) conv3x3s that convert input from branch j to branch i
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_channels

    def forward(self, x, injection, v=None, *args):
        """
        The two steps of a multiscale DEQ module : a per-resolution residual block and 
        a parallel multiscale fusion step.
        """
        if injection is None:
            injection = [0] * len(x)
        if self.num_branches == 1:
            return [self.branches[0](x[0], injection[0], v)]

        # Step 1: Per-resolution residual block (with additional context vector v for meta-learning tasks)
        x_block = []
        if v is not None:
            v = torch.chunk(v, self.num_branches, dim=1)
            for i in range(self.num_branches):
                x_block.append(self.branches[i](x[i], injection[i], v[i]))
        else:
            for i in range(self.num_branches):
                x_block.append(self.branches[i](x[i], injection[i]))
        
        # Step 2: Multiscale fusion
        x_fuse = []
        for i in range(self.num_branches):
            y = 0
            
            # Start fusing all #j -> #i up/down-samplings
            for j in range(self.num_branches):
                y += x_block[j] if i == j else self.fuse_layers[i][j](x_block[j])
            x_fuse.append(self.post_fuse_layers[i](y))
            
        return x_fuse

class MDEQBranchMerge(nn.Module):
    def __init__(self, cfg, num_channels, output_channels, **kwargs):
        """ 
        Module to merge equilibria of multiple resolutions and output an aggregated feature vector.
        The procedure is identical to the one in mdeq.MDEQClsNet
        """
        global BN_MOMENTUM
        super(MDEQBranchMerge, self).__init__()
        self.head_channels = cfg['MODEL']['EXTRA']['FULL_STAGE']['HEAD_CHANNELS']
        self.final_chansize = cfg['MODEL']['EXTRA']['FULL_STAGE']['FINAL_CHANSIZE']
        self.output_channels = output_channels

        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(num_channels)
        self.output_layer = nn.Linear(self.final_chansize, self.output_channels)
            
    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = self.head_channels
        
        # Increasing the number of channels on each resolution when doing classification. 
        incre_modules = []
        for i, channels  in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block, channels, head_channels[i], blocks=1, stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
            
        # Downsample the high-resolution streams
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
    
    def forward(self, y_list, train_step=0, **kwargs):
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i+1](y_list[i+1]) + self.downsamp_modules[i](y)
        y = self.final_layer(y)

        # Pool to a 1x1 vector (if needed)
        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
        y = self.output_layer(y)
        
        return y
    
    def _reset(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class MDEQBranchInterp(nn.Module):
    def __init__(self, cfg, num_channels, output_channels, **kwargs):
        """
        Build an MDEQ Segmentation model with the given hyperparameters
        """
        global BN_MOMENTUM
        super(MDEQBranchInterp, self).__init__()
        
        # Last layer
        last_inp_channels = np.int(np.sum(num_channels))
        if GNORM:
            normlayer = nn.GroupNorm(NUM_GROUPS, last_inp_channels, affine=True)
        else:
            normlayer = nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM)

        self.last_layer = nn.Sequential(nn.Conv2d(last_inp_channels, last_inp_channels, kernel_size=1),
                                        normlayer,
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(last_inp_channels, output_channels, 3, 
                                                  stride=1, padding=1))
               
    def forward(self, y_list, train_step=0, **kwargs):
        
        # Segmentation Head
        y0_h, y0_w = y_list[0].size(2), y_list[0].size(3)
        all_res = [y_list[0]]
        for y in y_list[1:]:
            all_res.append(F.interpolate(y, size=(y0_h, y0_w), mode='bilinear', align_corners=True))

        y = torch.cat(all_res, dim=1)
        all_res = None
        # torch.cuda.empty_cache()
        y = self.last_layer(y)
        return y
    
    def _reset(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.01)
            elif (isinstance(m, nn.BatchNorm2d))and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _copy(self, other):
        """
        Copy the parameter of an MDEQ layer. First copy the residual block, then the multiscale fusion part.
        """
    
        self.last_layer[0].weight.data = other.last_layer[0].weight.data.clone()
        self.last_layer[0].bias.data = other.last_layer[0].bias.data.clone()
        self.last_layer[3].weight.data = other.last_layer[3].weight.data.clone()
        self.last_layer[3].bias.data = other.last_layer[3].bias.data.clone()
        try:
            self.last_layer[1].weight.data = other.last_layer[1].weight.data.clone()
            self.last_layer[1].bias.data = other.last_layer[1].bias.data.clone()
            self.last_layer[1].running_mean.data = other.last_layer[1].running_mean.data.clone()
            self.last_layer[1].running_var.data = other.last_layer[1].running_var.data.clone()
        except:
            print("Did not set affine=True for gnorm(s)?")
        

class MDEQJIIOVAECore(nn.Module):
    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ Decoder (and an Encoder for MDEQVAE) with the given hyperparameters.
        """
        super(MDEQJIIOVAECore, self).__init__()
        global BN_MOMENTUM
        BN_MOMENTUM = kwargs.get("BN_MOMENTUM", 0.1)
        self.parse_cfg(cfg)
        init_chansize = self.init_chansize

        self.downsample = nn.Sequential(
            conv3x3(self.image_size[0], init_chansize, stride=(2 if self.downsample_times >= 1 else 1)),
            nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            conv3x3(init_chansize, init_chansize, stride=(2 if self.downsample_times >= 2 else 1)),
            nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        if GNORM:
            normlayer1 = nn.GroupNorm(NUM_GROUPS//2, init_chansize, affine=True)
            normlayer2 = nn.GroupNorm(NUM_GROUPS//2, init_chansize, affine=True)
        else:
            normlayer1 = nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM)
            normlayer2 = nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM)
        self.upsample = nn.Sequential(
            normlayer1,
            nn.ReLU(inplace=True),
            convTranspose3x3(init_chansize, init_chansize, stride=(2 if self.downsample_times >= 2 else 1)),
            normlayer2,
            nn.ReLU(inplace=True),
            convTranspose3x3(init_chansize, self.image_size[0], stride=(2 if self.downsample_times >= 1 else 1)),
        )

        downsample_factor = 2 ** self.downsample_times
        assert np.all([d % downsample_factor == 0 for d in self.image_size[1:]]), \
               "Image size must be divisible by 2^downsample_times"
        init_hw = [d // downsample_factor for d in self.image_size[1:]]
        branches_hw = [init_hw]
        for i in range(self.num_branches - 1):
            branches_hw.append([d // 2 for d in branches_hw[-1]])

        self.branch_shapes = [[self.num_channels[i], h, w] for i, (h, w) in enumerate(branches_hw)]

        # Maps from vae latent space to decoder inputs (multiple resolutions)
        self.latent_maps = nn.ModuleList(
            [nn.Linear(self.encoding_channels, np.prod(s)) for s in self.branch_shapes]
        )

        self.fullstage_cfg = cfg['MODEL']['EXTRA']['FULL_STAGE']      
        # Defines the encoder and decoder wrappers to perform forward pass through encoder/decoder
        self.encoder_stage0, self.encoder, self.encoder_cpy, self.encoder_deq = self._make_mdeq(self.dropout)
        self.decoder_stage0, self.decoder, self.decoder_cpy, self.decoder_deq = self._make_mdeq(0.)

        # Wrapper for input optimization with decoder.
        self.enc_dec_deq = MDEQOptWrapper(self.encoder, self.encoder_cpy, self.decoder, self.decoder_cpy)

        self.branch_merge = MDEQBranchMerge(cfg, self.num_channels, self.encoding_channels * 2)
        self.branch_interp = MDEQBranchInterp(cfg, self.num_channels, init_chansize)
        if cfg['MODEL']['EXTRA']['FULL_BWD']:
            self.latent_maps_copy = copy.deepcopy(self.latent_maps)
            self.branch_interp_copy = copy.deepcopy(self.branch_interp)
            self.upsample_copy = copy.deepcopy(self.upsample)
        self.cfg = cfg
    def define_opt_wrapper(self,):
        """
        Function used when using a MDEQVAE model to perform JIIO
        """
        # Defines the JIIO wrapper for MDEQVAE model.
        self.enc_dec_deq = MDEQOptWrapper(self.encoder, self.encoder_cpy, self.decoder, self.decoder_cpy)

    def _make_mdeq(self, dropout):
        """
        Construct the mdeq module
        """
        # Input injection module
        if self.downsample_times == 0 and self.num_branches <= 2:
            stage0 = None
        else:
            stage0 = nn.Sequential(nn.Conv2d(self.init_chansize, self.init_chansize, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.init_chansize, momentum=BN_MOMENTUM),
                                   nn.ReLU(False))

        fullstage = self._make_stage(self.fullstage_cfg, self.num_channels, dropout=dropout)
        fullstage_cpy = copy.deepcopy(fullstage)

        if self.wnorm:
            fullstage._wnorm()

        for param in fullstage_cpy.parameters():
            param.requires_grad_(True)
        deq = MDEQWrapper(fullstage, fullstage_cpy)
        return stage0, fullstage, fullstage_cpy, deq

    def parse_cfg(self, cfg):
        global DEQ_EXPAND, NUM_GROUPS, GNORM
        self.num_branches = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_BRANCHES']
        self.num_channels = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_CHANNELS']
        self.init_chansize = self.num_channels[0]
        self.num_layers = cfg['MODEL']['NUM_LAYERS']
        self.dropout = cfg['MODEL']['DROPOUT']
        self.wnorm = cfg['MODEL']['WNORM']
        self.f_thres = cfg['MODEL']['F_THRES']
        self.b_thres = cfg['MODEL']['B_THRES']
        self.num_classes = cfg['MODEL']['NUM_CLASSES']
        self.downsample_times = cfg['MODEL']['DOWNSAMPLE_TIMES']
        self.pretrain_steps = cfg['TRAIN']['PRETRAIN_STEPS']
        self.encoding_channels = cfg['MODEL']['EXTRA']["FULL_STAGE"]["ENCODING_CHANNELS"]
        self.image_size = cfg['MODEL']['IMAGE_SIZE']
        self.output_nl = cfg['MODEL']['OUTPUT_NL']
        self.mem = cfg['MODEL']['MEM']
        self.alpha_0 = cfg['MODEL']['ALPHA_0']
        self.traj_reg = cfg['MODEL']['TRAJ_REG']
        self.jiio_thres = cfg['MODEL']['JIIO_THRES']
        self.inv_prob = cfg['TRAIN']['INV_PROB']
        DEQ_EXPAND = cfg['MODEL']['EXPANSION_FACTOR']
        NUM_GROUPS = cfg['MODEL']['NUM_GROUPS']
        GNORM = cfg['MODEL']['GNORM']

    def _make_stage(self, layer_config, num_channels, dropout=0.0):
        """
        Build an MDEQ block with the given hyperparameters
        """
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        block_type = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        
        return MDEQModule(num_branches, block_type, num_blocks, num_channels, fuse_method, dropout=dropout)

    def _reparameterize(self, mu, sigma):
        return torch.randn_like(mu) * sigma + mu

    def _encode(self, x, train_step, **kwargs):
        """
        Forward pass through just the encoder
        """
        x = self.downsample(x)
        f_thres = kwargs.get('f_thres', self.f_thres)
        b_thres = kwargs.get('b_thres', self.b_thres)
        writer = kwargs.get('writer', None)     # For tensorboard
        dev = x.device

        # Inject only to the highest resolution...
        x_list = [self.encoder_stage0(x) if self.encoder_stage0 else x]
        for i in range(1, self.num_branches):
            bsz, _, H, W = x_list[-1].shape
            x_list.append(torch.zeros(bsz, self.num_channels[i], H//2, W//2).to(dev))   # ... and the rest are all zeros
            
        z_list = [torch.zeros_like(elem) for elem in x_list]
        
        # For variational dropout mask resetting and weight normalization re-computations
        self.encoder._reset(z_list)
        self.encoder_cpy._copy(self.encoder)
        
        # Multiscale Deep Equilibrium Decoder!
        if 0 <= train_step < self.pretrain_steps:
            for layer_ind in range(self.num_layers):
                z_list = self.encoder(z_list, x_list)
        else:
            if train_step == self.pretrain_steps:
                torch.cuda.empty_cache()
            z_list = self.encoder_deq(z_list, x_list, threshold=f_thres, train_step=train_step, writer=writer)
        self.z_list = z_list
        merged_z = self.branch_merge(z_list)
        mu, logsd = torch.split(merged_z, merged_z.size(1) // 2, dim=1)
        return mu, logsd

    def _decode(self, latent, train_step, **kwargs):
        """
        Forward pass through just the Decoder
        """
        f_thres = kwargs.get('f_thres', self.f_thres)
        b_thres = kwargs.get('b_thres', self.b_thres)
        writer = kwargs.get('writer', None)     # For tensorboard
        dev = latent.device

        # Inject to all resolutions
        x_prime_list = [self.latent_maps[i](latent).view(-1, *s) for i, s in enumerate(self.branch_shapes)]

        z_prime_list = [torch.zeros_like(elem) for elem in x_prime_list]
        z_prime_list = kwargs.get('z_prime_list', z_prime_list)

        # For variational dropout mask resetting and weight normalization re-computations
        self.decoder._reset(z_prime_list)
        self.decoder_cpy._copy(self.decoder)

        # Multiscale Deep Equilibrium Decoder!
        if 0 <= train_step < self.pretrain_steps:
            for layer_ind in range(self.num_layers):
                z_prime_list = self.decoder(z_prime_list, x_prime_list)
        else:
            if train_step == self.pretrain_steps:
                torch.cuda.empty_cache()
            z_prime_list = self.decoder_deq(z_prime_list, x_prime_list, threshold=f_thres, train_step=train_step, writer=writer)
        self.x_prime_list = z_prime_list
        aggregated = self.branch_interp(z_prime_list)
        output = self.upsample(aggregated)
        if self.output_nl=='sigmoid':
            output = torch.sigmoid(output)
        elif self.output_nl=='tanh':
            output = torch.tanh(output)
        return output

    def _forward(self, x, train_step=-1, **kwargs):
        mu, logsd = self._encode(x, train_step, **kwargs)
        sampled_latent = self._reparameterize(mu, logsd.exp())
        output = self._decode(sampled_latent, train_step, **kwargs)

        return mu, logsd, output

    def _deq_constr_opt(self, x=None, latent=None, x_init=None, train_step=-1, **kwargs):
        """
        Perform input/latent optimization with JIIO on MDEQ Decoder.
        Returns : ouput, 
                  traj_z (samples of z used to compute jac_loss) 
                  traj_u (samples of input used to compute jac_loss)
                  jac_loss (Jacobian trace of f computed using the Hutchinson estimator)
        
        """
        f_thres = kwargs.get('f_thres', self.f_thres)
        b_thres = kwargs.get('b_thres', self.b_thres)
        writer = kwargs.get('writer', None)     # For tensorboard

        # If latents are provided but the target image is not, then just perform 
        # forward pass throught the decoder
        if latent is not None and x is None:
            output = self._decode(latent, train_step)
            return output

        # If an initialization for the latents is not provided, initialize to zero
        if latent is None:
            latent = torch.zeros((x.shape[0], self.encoding_channels), device=x.device, dtype=x.dtype)
        if x_init is None:
            x_init = torch.zeros_like(x)
        self.xgt = x.clone()
        self.latent = latent

        x_prime_list = [self.latent_maps[i](latent).view(-1, *s) for i, s in enumerate(self.branch_shapes)]
        z_prime_list = [torch.zeros_like(elem) for elem in x_prime_list]

        self.decoder._reset(z_prime_list)
        self.decoder_cpy._copy(self.decoder)
        if self.cfg['MODEL']['EXTRA']['FULL_BWD']:
            self._copy_fn()

        # Perform JIIO!        
        x_list = torch.zeros_like(latent)
        new_z = self.enc_dec_deq(x_list, z_prime_list, x, self.dec_output_layer, self.get_xprimelist, threshold=f_thres, train_step=train_step, writer=writer, mem=self.mem, alpha_0=self.alpha_0, jiio_thres=self.jiio_thres, traj_reg=self.traj_reg, inv_prob=self.inv_prob)

        # Project z onto the output
        aggregated = self.branch_interp(new_z)
        output = self.upsample(aggregated)
        if self.output_nl=='sigmoid':
            output = torch.sigmoid(output)
        elif self.output_nl=='tanh':
            output = torch.tanh(output)

        return output, self.enc_dec_deq.z_traj, self.enc_dec_deq.u_traj, self.enc_dec_deq.jac_loss

    def dec_output_layer(self, z, cutoffs, bwd_pass):
        """
        Project z to output : h(z) in the paper
        """
        z_prime_list = DEQFunc2d.vec2list(z, cutoffs)
        if bwd_pass:
            aggregated = self.branch_interp_copy(z_prime_list)
            output = self.upsample_copy(aggregated)
        else:
            aggregated = self.branch_interp(z_prime_list)
            output = self.upsample(aggregated)

        if self.output_nl=='sigmoid':
            output = torch.sigmoid(output)
        elif self.output_nl=='tanh':
            output = torch.tanh(output)
        return output

    def get_xprimelist(self, x, cutoffs, bwd_pass):
        """
        Compute input injection to each branch
        """
        mu = x

        if bwd_pass:
            x_list_prime = [self.latent_maps_copy[i](mu).view(-1, *s) for i, s in enumerate(self.branch_shapes)]
        else:
            x_list_prime = [self.latent_maps[i](mu).view(-1, *s) for i, s in enumerate(self.branch_shapes)]
        return x_list_prime

    def _copy_fn(self):
        """
        Copy the input and output layers if performing full backward
        """
        for i in range(len(self.branch_shapes)):
            self.latent_maps_copy[i].weight.data = self.latent_maps[i].weight.data
            self.latent_maps_copy[i].bias.data = self.latent_maps[i].bias.data

        self.branch_interp_copy._copy(self.branch_interp)
        
        self.upsample_copy[2].weight.data = self.upsample[2].weight.data.clone()
        self.upsample_copy[5].weight.data = self.upsample[5].weight.data.clone()
        try:
            self.upsample_copy[0].weight.data = self.upsample[0].weight.data.clone()
            self.upsample_copy[0].bias.data = self.upsample[0].bias.data.clone()
            self.upsample_copy[0].running_mean.data = self.upsample[0].running_mean.data.clone()
            self.upsample_copy[0].running_var.data = self.upsample[0].running_var.data.clone()
            self.upsample_copy[3].weight.data = self.upsample[3].weight.data.clone()
            self.upsample_copy[3].bias.data = self.upsample[3].bias.data.clone()
            self.upsample_copy[3].running_mean.data = self.upsample[3].running_mean.data.clone()
            self.upsample_copy[3].running_var.data = self.upsample[3].running_var.data.clone()
        except:
            print("Did not set affine=True for gnorm(s)?")
        

    def forward(self, x, train_step=-1, **kwargs):
        raise NotImplementedError
            

class MDEQNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ model with the given hyperparameters
        """
        super(MDEQNet, self).__init__()
        global BN_MOMENTUM
        BN_MOMENTUM = kwargs.get('BN_MOMENTUM', 0.1)
        self.parse_cfg(cfg)
        init_chansize = self.init_chansize


        if GNORM:
            normlayer1 = nn.GroupNorm(NUM_GROUPS//2, init_chansize, affine=True)
            normlayer2 = nn.GroupNorm(NUM_GROUPS//2, init_chansize, affine=True)
        else:
            normlayer1 = nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM)
            normlayer2 = nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM)

        self.downsample = nn.Sequential(
            conv3x3(self.image_size[0], self.init_chansize, stride=(2 if self.downsample_times >= 1 else 1)),
            normlayer1,
            nn.ReLU(inplace=True),
            conv3x3(init_chansize, init_chansize, stride=(2 if self.downsample_times >= 2 else 1)),
            normlayer2,
            nn.ReLU(inplace=True)
        )
        
        # PART I: Input injection module
        if self.downsample_times == 0 and self.num_branches <= 2:
            # We use the downsample module above as the injection transformation
            self.stage0 = None
        else:
            if GNORM:
                normlayer3 = nn.GroupNorm(NUM_GROUPS//2, init_chansize, affine=True)
            else:
                normlayer3 = nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM)
            self.stage0 = nn.Sequential(nn.Conv2d(self.init_chansize, self.init_chansize, kernel_size=1, bias=False),
                                        normlayer3,
                                        nn.ReLU(False))

        downsample_factor = 2 ** self.downsample_times
        assert np.all([d % downsample_factor == 0 for d in self.image_size[1:]]), \
               "Image size must be divisible by 2^downsample_times"
        init_hw = [d // downsample_factor for d in self.image_size[1:]]
        branches_hw = [init_hw]
        for i in range(self.num_branches - 1):
            branches_hw.append([d // 2 for d in branches_hw[-1]])

        self.branch_shapes = [[self.num_channels[i], h, w] for i, (h, w) in enumerate(branches_hw)]
        
        # PART II: MDEQ's f_\theta layer
        self.fullstage_cfg = cfg['MODEL']['EXTRA']['FULL_STAGE']      
        num_channels = self.num_channels
        block = blocks_dict[self.fullstage_cfg['BLOCK']]
        self.fullstage = self._make_stage(self.fullstage_cfg, num_channels, dropout=self.dropout)
        self.fullstage_copy = copy.deepcopy(self.fullstage)
        
        if self.wnorm:
            self.fullstage._wnorm()
            
        for param in self.fullstage_copy.parameters():
            param.requires_grad_(False)

        # The original DEQ layer wrapper
        self.deq = MDEQClsWrapper(self.fullstage, self.fullstage_copy)

        # Wrapper to perform JIIO (to compute adversarial examples and corresponding output)
        self.deqproxy = MDEQWrapper(self.fullstage, self.fullstage_copy)
        self.iodrop = VariationalHidDropout2d(0.0)

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
            if GNORM:
                normlayer = nn.GroupNorm(NUM_GROUPS//2, out_channels, affine=True)
            else:
                normlayer = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            downsamp_module = nn.Sequential(conv3x3(in_channels, out_channels, stride=2, bias=True),
                                            normlayer,
                                            nn.ReLU(inplace=True))
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        if GNORM:
            normlayer = nn.GroupNorm(NUM_GROUPS//2, self.final_chansize, affine=True)
        else:
            normlayer = nn.BatchNorm2d(self.final_chansize, momentum=BN_MOMENTUM)
        # Final FC layers
        final_layer = nn.Sequential(nn.Conv2d(head_channels[len(pre_stage_channels)-1] * head_block.expansion,
                                              self.final_chansize,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0),
                                    normlayer,
                                    nn.ReLU(inplace=True))
        return incre_modules, downsamp_modules, final_layer

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            if GNORM:
                normlayer = nn.GroupNorm(NUM_GROUPS, planes * block.expansion, affine=True)
            else:
                normlayer = nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM)
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                normlayer)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)
    
    def parse_cfg(self, cfg):
        global DEQ_EXPAND, NUM_GROUPS, GNORM
        self.num_branches = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_BRANCHES']
        self.num_channels = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_CHANNELS']
        self.init_chansize = self.num_channels[0]
        self.num_layers = cfg['MODEL']['NUM_LAYERS']
        self.dropout = cfg['MODEL']['DROPOUT']
        self.wnorm = cfg['MODEL']['WNORM']
        self.f_thres = cfg['MODEL']['F_THRES']
        self.b_thres = cfg['MODEL']['B_THRES']
        self.num_classes = cfg['MODEL']['NUM_CLASSES']
        self.downsample_times = cfg['MODEL']['DOWNSAMPLE_TIMES']
        self.pretrain_steps = cfg['TRAIN']['PRETRAIN_STEPS']
        self.image_size = cfg['MODEL']['IMAGE_SIZE']
        self.epsilon = cfg['MODEL']['EPSILON']
        self.mem = cfg['MODEL']['MEM']
        self.alpha_0 = cfg['MODEL']['ALPHA_0']
        self.traj_reg = cfg['MODEL']['TRAJ_REG']
        self.jiio_thres = cfg['MODEL']['JIIO_THRES']
        DEQ_EXPAND = cfg['MODEL']['EXPANSION_FACTOR']
        NUM_GROUPS = cfg['MODEL']['NUM_GROUPS']
        GNORM = cfg['MODEL']['GNORM']
        
    def _make_stage(self, layer_config, num_channels, dropout=0.0):
        """
        Build an MDEQ block with the given hyperparameters
        """
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        block_type = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        
        return MDEQModule(num_branches, block_type, num_blocks, num_channels, fuse_method, dropout=dropout)
    
    def _forward(self, x, y=None, train_step=-1, **kwargs):
        """
        Perform input optimization using JIIO if y is given 
        or perform vanilla forward pass if y is not given
        Return : output label y_hat
                 jac_loss (Jacobian loss : Jacobian trace of f computed using the Hutchinson estimator)
        """
        num_branches = self.num_branches
        f_thres = kwargs.get('f_thres', self.f_thres)
        b_thres = kwargs.get('b_thres', self.b_thres)
        writer = kwargs.get('writer', None)     # For tensorboard
        proj_adam = kwargs.get('proj_adam', False)
        self.x_shape = x.shape
        dev = x.device
        if y is None:
            x = self.downsample(x)
            
            # Inject only to the highest resolution...
            x_list = [self.stage0(x) if self.stage0 else x]
            for i in range(1, num_branches):
                bsz, _, H, W = x_list[-1].shape
                x_list.append(torch.zeros(bsz, self.num_channels[i], H//2, W//2).to(dev))   # ... and the rest are all zeros
            z_list = [torch.zeros_like(elem) for elem in x_list]
            self.fullstage._reset(z_list)
            self.fullstage_copy._copy(self.fullstage)
            z_list = self.deqproxy(z_list, x_list, threshold=f_thres, train_step=train_step, writer=writer)
        
            y_list = self.iodrop(z_list)
            y_hat = self.output_layer(y_list)

            return y_hat, self.deqproxy.jac_loss
        else:
            x = x.reshape(x.shape[0], -1)
            z_list = [torch.zeros([x.shape[0],] + s).to(x) for s in self.branch_shapes]
            
            # For variational dropout mask resetting and weight normalization re-computations
            self.fullstage._reset(z_list)
            self.fullstage_copy._copy(self.fullstage)
            
            # Multiscale Deep Equilibrium!
            z_list = self.deq(x, z_list, y, self.output_layer, self.get_xlist, threshold=f_thres, train_step=train_step, writer=writer, proj_adam=proj_adam, epsilon=self.epsilon, mem=self.mem, alpha_0=self.alpha_0, traj_reg=self.traj_reg, jiio_thres=self.jiio_thres)
        
            y_list = self.iodrop(z_list)
            y_hat = self.output_layer(y_list)

            return y_hat, self.deq.jac_loss
    
    def output_layer(self, z, cutoffs=None, bwd_pass=False):
        """
        computes the output y : Computes y = h(z)
        """
        if cutoffs is None:
            z_list = z
        else:
            z_list = DEQFunc2d.vec2list(z, cutoffs)
        # Classification Head
        y = self.incre_modules[0](z_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i+1](z_list[i+1]) + self.downsamp_modules[i](y)
        y = self.final_layer(y)

        # Pool to a 1x1 vector (if needed)
        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
        y = self.classifier(y)
        return y


    def get_xlist(self, x, cutoffs, bwd_pass=False):
        """
        Computes input injection for each branch given inputs x
        """
        x = self.downsample(x.view(self.x_shape))
        dev = x.device
        
        # Inject only to the highest resolution...
        x_list = [self.stage0(x) if self.stage0 else x]
        for i in range(1, self.num_branches):
            bsz, _, H, W = x_list[-1].shape
            x_list.append(torch.zeros(bsz, self.num_channels[i], H//2, W//2).to(dev))   # ... and the rest are all zeros
        return x_list


    def forward(self, x, train_step=-1, **kwargs):
        # raise NotImplemented    # To be inherited & implemented by MDEQClsNet and MDEQSegNet (see mdeq.py)
        return self._forward(x, train_step, **kwargs)

class MDEQIMAMLCore(nn.Module):

    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ model with the given hyperparameters. This MDEQ model takes an additional 
        context vector v as input which identifies the specific task. 
        """
        super(MDEQIMAMLCore, self).__init__()
        global BN_MOMENTUM
        BN_MOMENTUM = kwargs.get('BN_MOMENTUM', 0.1)
        self.parse_cfg(cfg)
        init_chansize = self.init_chansize

        self.downsample = nn.Sequential(
            conv3x3(self.image_size[0], init_chansize, stride=(2 if self.downsample_times >= 1 else 1)),
            nn.GroupNorm(NUM_GROUPS//2, init_chansize, affine=True),
            nn.ReLU(inplace=True),
        )
        
        self.stage0 = None
        self.fullstage_cfg = cfg['MODEL']['EXTRA']['FULL_STAGE']      
        num_channels = self.num_channels
        block = blocks_dict[self.fullstage_cfg['BLOCK']]
        self.fullstage = self._make_stage(self.fullstage_cfg, num_channels, dropout=self.dropout)
        self.fullstage_copy = copy.deepcopy(self.fullstage)
        
        if self.wnorm:
            self.fullstage._wnorm()
            
        for param in self.fullstage_copy.parameters():
            param.requires_grad_(False)

        # Defining the MDEQIMAML wrapper which performs JIIO/PGD on context vector v
        self.deq = MDEQIMAMLWrapper(self.fullstage, self.fullstage_copy)
        self.iodrop = VariationalHidDropout2d(0.0)

        downsample_factor = 2 ** self.downsample_times
        assert np.all([d % downsample_factor == 0 for d in self.image_size[1:]]), \
               "Image size must be divisible by 2^downsample_times"
        init_hw = [d // downsample_factor for d in self.image_size[1:]]
        branches_hw = [init_hw]
        self.concat = False
        for i in range(self.num_branches - 1):
            branches_hw.append([d // 2 for d in branches_hw[-1]])

        self.branch_shapes = [[self.num_channels[i], h, w] for i, (h, w) in enumerate(branches_hw)]
        
        self.head_channels = cfg['MODEL']['EXTRA']['FULL_STAGE']['HEAD_CHANNELS']
        self.final_chansize = cfg['MODEL']['EXTRA']['FULL_STAGE']['FINAL_CHANSIZE']

        # Classification Head
        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(self.num_channels)#self.final_chansize)
        if self.num_branches==1:
            self.classifier = nn.Linear(self.num_channels[0]*4*4, self.num_classes)
        elif self.num_branches == 2:
            self.classifier = nn.Linear(self.num_channels[0]*(4*4 + 2*2*2), self.num_classes)
            
        self.final_layer_copy = copy.deepcopy(self.final_layer)
        self.classifier_copy = copy.deepcopy(self.classifier)
        self.cfg = cfg
        
    def parse_cfg(self, cfg):
        global DEQ_EXPAND, NUM_GROUPS, GNORM, ENCODING_CHANNELS
        self.num_branches = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_BRANCHES']
        self.num_channels = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_CHANNELS']
        self.init_chansize = self.num_channels[0]
        self.num_layers = cfg['MODEL']['NUM_LAYERS']
        self.dropout = cfg['MODEL']['DROPOUT']
        self.wnorm = cfg['MODEL']['WNORM']
        self.f_thres = cfg['MODEL']['F_THRES']
        self.b_thres = cfg['MODEL']['B_THRES']
        self.num_classes = cfg['IMAML']['N_WAY']
        self.downsample_times = cfg['MODEL']['DOWNSAMPLE_TIMES']
        self.pretrain_steps = cfg['TRAIN']['PRETRAIN_STEPS']
        ENCODING_CHANNELS = self.encoding_channels = cfg['MODEL']['EXTRA']["FULL_STAGE"]["ENCODING_CHANNELS"]
        self.image_size = cfg['MODEL']['IMAGE_SIZE']
        self.mem = cfg['MODEL']['MEM']
        self.alpha_0 = cfg['MODEL']['ALPHA_0']
        self.traj_reg = cfg['MODEL']['TRAJ_REG']
        self.jiio_thres = cfg['MODEL']['JIIO_THRES']
        DEQ_EXPAND = cfg['MODEL']['EXPANSION_FACTOR']
        NUM_GROUPS = cfg['MODEL']['NUM_GROUPS']
        GNORM = True
            
    def _make_stage(self, layer_config, num_channels, dropout=0.0):
        """
        Build an MDEQ block with the given hyperparameters
        """
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        block_type = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        
        return MDEQModule(num_branches, block_type, num_blocks, num_channels, fuse_method, dropout=dropout)
    
    def _forward(self, x_tr, y_tr, x_te, train_step=-1, **kwargs):
        """
        Perform PGD/JIIO on context vectors v to identify the task using training samples (x_tr, y_tr)
        and then compute the labels/outputs for the test samples x_te.
        """
        num_branches = self.num_branches
        f_thres = kwargs.get('f_thres', self.f_thres)
        b_thres = kwargs.get('b_thres', self.b_thres)
        writer = kwargs.get('writer', None)     # For tensorboard
        proj_adam = kwargs.get('proj_adam', False)
        num_tasks, bsz_tr = x_tr.shape[:2]
        bsz_te = x_te.shape[1]
        x_tr = x_tr.reshape(-1, *x_tr.shape[2:])
        x_te = x_te.reshape(-1, *x_te.shape[2:])
        y_tr = y_tr.reshape(-1, *y_tr.shape[2:])
        x_tr = self.downsample(x_tr)
        x_te = self.downsample(x_te)
        dev = x_tr.device
        
        # Inject only to the highest resolution...
        x_tr_list = [self.stage0(x_tr) if self.stage0 else x_tr]
        x_te_list = [self.stage0(x_te) if self.stage0 else x_te]
        for i in range(1, num_branches):
            bsz, _, H, W = x_tr_list[-1].shape
            x_tr_list.append(torch.zeros(bsz, self.num_channels[i], H//2, W//2).to(dev))   # ... and the rest are all zeros
            x_te_list.append(torch.zeros(bsz, self.num_channels[i], H//2, W//2).to(dev))   # ... and the rest are all zeros

        v = torch.zeros((num_tasks, self.encoding_channels), device=x_tr.device, dtype=x_tr.dtype)
        self.latent = v

        z_tr_list = [torch.zeros_like(elem) for elem in x_tr_list]
        z_te_list = [torch.zeros_like(elem) for elem in x_te_list]
        
        # For variational dropout mask resetting and weight normalization re-computations
        self.fullstage._reset(z_tr_list)
        self.fullstage_copy._copy(self.fullstage)
        self._copy_fn()
        
        # Perform JIIO/PGD on v and then compute z_te for each x_te!
        z_te = self.deq(z_tr_list, z_te_list, v, x_tr_list, x_te_list, y_tr, self.output_layer, bsz_tr, bsz_te, self.final_layer_copy, self.classifier_copy, threshold=f_thres, train_step=train_step, writer=writer, proj_adam=proj_adam, mem=self.mem, alpha_0=self.alpha_0, traj_reg=self.traj_reg, jiio_thres=self.jiio_thres)
        
        y_hat = self.output_layer(z_te, self.deq.z_cutoffs, False)
        return y_hat, self.deq.jac_loss

    def output_layer(self, z, cutoffs, bwd_pass):
        """
        Output layer : Computes y = h(z)
        """
        y_list = DEQFunc2d.vec2list(z, cutoffs)        
        # Classification Head
        if bwd_pass:
            if self.num_branches == 1:
                y = self.final_layer_copy(y_list[0])
            elif self.num_branches > 1:
                y_list1 = [self.final_layer_copy[i](y_list[i]) for i in range(self.num_branches)]
                y = torch.cat(y_list1, dim=-1)
            y = self.classifier_copy(y)
        else:
            if self.num_branches == 1:
                y = self.final_layer(y_list[0])
            elif self.num_branches > 1:
                y_list1 = [self.final_layer[i](y_list[i]) for i in range(self.num_branches)]
                y = torch.cat(y_list1, dim=-1)
            y = self.classifier(y)
        return y


    def _copy_fn(self):
        """
        Copies the output layers : only used when performing Full Backward
        """
        if self.num_branches == 1:
            self.final_layer_copy[0].weight.data = self.final_layer[0].weight.data.clone()
            self.final_layer_copy[0].bias.data = self.final_layer[0].bias.data.clone()
        elif self.num_branches > 1:
            for j in range(len(self.final_layer_copy)):
                self.final_layer_copy[j][0].weight.data = self.final_layer[j][0].weight.data.clone()
                self.final_layer_copy[j][0].bias.data = self.final_layer[j][0].bias.data.clone()
        self.classifier_copy.weight.data = self.classifier.weight.data.clone()
        self.classifier_copy.bias.data = self.classifier.bias.data.clone()

    def _make_head(self, pre_stage_channels):
        """
        Create a simple classification head that:
           - downsamples the outputs and concats.
        """
        head_block = Bottleneck
        d_model = self.init_chansize
        head_channels = self.head_channels
        if len(pre_stage_channels)==1:
            incre_modules = nn.ModuleList([])
            downsamp_modules = nn.ModuleList([])
            final_layer = nn.Sequential(
                                    nn.GroupNorm(NUM_GROUPS, pre_stage_channels[0], affine=True),
                                    nn.AvgPool2d(7,7),
                                    nn.Flatten()
                                    )
            return incre_modules, downsamp_modules, final_layer

        elif len(pre_stage_channels)>1:
            incre_modules = nn.ModuleList([])
            downsamp_modules = nn.ModuleList([])
            final_layer = []
            for j, channels in enumerate(pre_stage_channels):
                final_layer.append(nn.Sequential(
                                    nn.GroupNorm(NUM_GROUPS, channels, affine=True),
                                    nn.AvgPool2d(7,7),
                                    nn.Flatten()
                                    ))
            final_layer = nn.ModuleList(final_layer)
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
    
    def forward(self, x, train_step=-1, **kwargs):
        # raise NotImplemented    # To be inherited (see mdeq.py)
        return self._forward(x, train_step, **kwargs)



