# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import sys
# import ipdb

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter

import _init_paths
import models
from config import config
from config import update_config
from core.imaml_function import train
from utils.modelsummary import get_model_summary
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from termcolor import colored
from natsort import natsorted
from PIL import Image
import visdom
from datasets import dataset
import pickle
import numpy as np

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')
    parser.add_argument('--percent',
                        help='percentage of training data to use',
                        type=float,
                        default=1.0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--local_rank', 
                        help='Local process rank.',
                        type=int, 
                        default=-1, 
                        metavar='N') 
    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    viz = None
    args = parse_args()
    print(colored("Setting default tensor type to cuda.FloatTensor", "cyan"))
    torch.multiprocessing.set_start_method('spawn')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))
    
    rank = config.RANK

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    
    if config.DATASET.DATASET == 'sinusoid':
        model = eval('models.'+config.MODEL.NAME+'.get_imaml_fc')(config).cuda()    
    else:
        model = eval('models.'+config.MODEL.NAME+'.get_imaml')(config).cuda()

    dump_input = torch.rand(config.TRAIN.BATCH_SIZE_PER_GPU, config.IMAML.K_SHOT*config.IMAML.N_WAY, config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[2]).cuda()
    dump_y = torch.randint(low=0, high=config.IMAML.N_WAY, size=(config.TRAIN.BATCH_SIZE_PER_GPU, config.IMAML.K_SHOT*config.IMAML.N_WAY)).cuda()
    
    if config.TRAIN.MODEL_FILE:
        model.load_state_dict(torch.load(config.TRAIN.MODEL_FILE))
        logger.info(colored('=> loading model from {}'.format(config.TRAIN.MODEL_FILE), 'red'))
    
    if config.TEST.MODEL_FILE:
        try:
            model.load_state_dict(torch.load(config.TEST.MODEL_FILE))#, map_location=torch.device('cpu')))
        except:
            model.load_state_dict(torch.load(config.TEST.MODEL_FILE)['state_dict'])
        logger.info(colored('=> loading model from {}'.format(config.TEST.MODEL_FILE), 'red'))

    # copy lib files to save dir
    if rank == 0:
        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, 'models')
        if os.path.exists(models_dst_dir):
            shutil.rmtree(models_dst_dir)
        shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)
        models_dst_dir = os.path.join(final_output_dir, 'lib')
        if os.path.exists(models_dst_dir):
            shutil.rmtree(models_dst_dir)
        shutil.copytree(os.path.join(this_dir, '../lib'), models_dst_dir)

    if rank==0:
        writer_dict = {
            'writer': SummaryWriter(log_dir=tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }
    else:
        writer_dict = None

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    print("Finished constructing model!")

    # cross entropy
    if config.DATASET.DATASET == 'sinusoid':
        criterion = lambda output, target: ((output - target)**2).mean()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    

    optimizer = get_optimizer(config, model)
    lr_scheduler = None

    best_perf = 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint1000.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            model.module.load_state_dict(checkpoint['state_dict'])
            
            # Update weight decay if needed
            checkpoint['optimizer']['param_groups'][0]['weight_decay'] = config.TRAIN.WD
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            if 'lr_scheduler' in checkpoint:
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1e5, 
                                  last_epoch=checkpoint['lr_scheduler']['last_epoch'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            best_model = True
    # Data loading code
    dataset_name = config.DATASET.DATASET

    if dataset_name == 'mini-imagenet':
        traindir = os.path.join(config.DATASET.ROOT+'/images')
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_valid = transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])
        
        train_dataset = miniimagenet(traindir, shots=1, ways=5, meta_train=True, transform=transform_train, download=True)
      
    elif dataset_name == 'omniglot':
        task_defs = pickle.load(open(config.IMAML.LOAD_TASKS, 'rb'))
        task_defs = task_defs[:config.IMAML.NUM_TASKS]
        train_dataset = dataset.OmniglotFewShotDataset(task_defs=task_defs, img_size=config.MODEL.IMAGE_SIZE[1], GPU=False)

    elif dataset_name == 'sinusoid':
        train_dataset = dataset.SinusoidDataset(num_tasks=100000, GPU=False)

    sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, generator=torch.Generator(device='cuda'))

    if dataset_name == 'mini-imagenet':
        train_loader = BatchMetaDataLoader(train_dataset,
                             batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
                             shuffle=True,
                             num_workers=1)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
            sampler=sampler,
            shuffle=sampler is None,
            drop_last=True,
        )
    # Learning rate scheduler
    if lr_scheduler is None:
        if config.TRAIN.LR_SCHEDULER != 'step':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, len(train_loader)*config.TRAIN.END_EPOCH, eta_min=1e-6)
        elif isinstance(config.TRAIN.LR_STEP, list):
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
                last_epoch-1)
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
                last_epoch-1)
    niter = [0]

    # Training code
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
           
        # train for one epoch
        train(rank, config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
              final_output_dir, tb_log_dir, writer_dict, niter, args, visualizer=viz)
        torch.cuda.empty_cache()
        
    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()

