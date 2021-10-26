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
import numpy as np

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
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

import _init_paths
import models
from config import config
from config import update_config
from core.imaml_function import train, test
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
import ipdb

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

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    seed = 1265
    torch.manual_seed(seed)
    np.random.seed(seed)
    args = parse_args()
    print(colored("Setting default tensor type to cuda.FloatTensor", "cyan"))
    torch.multiprocessing.set_start_method('spawn')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    
    
    if config.DATASET.DATASET == 'sinusoid':
        model = eval('models.'+config.MODEL.NAME+'.get_imaml_fc')(config).cuda()    
    else:
        model = eval('models.'+config.MODEL.NAME+'.get_imaml')(config).cuda()

    if config.TRAIN.MODEL_FILE:
        model.load_state_dict(torch.load(config.TRAIN.MODEL_FILE))
        print(colored('=> loading model from {}'.format(config.TRAIN.MODEL_FILE), 'red'))
    elif config.TEST.MODEL_FILE:
        try:
            model.load_state_dict(torch.load(config.TEST.MODEL_FILE))#, map_location=torch.device('cpu')))
        except:
            model.load_state_dict(torch.load(config.TEST.MODEL_FILE)['state_dict'])
        print(colored('=> loading model from {}'.format(config.TEST.MODEL_FILE), 'red'))

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    # print("Finished constructing model!")

    if config.DATASET.DATASET == 'sinusoid':
        criterion = lambda output, target: ((output - target)**2).mean()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Data loading code
    dataset_name = config.DATASET.DATASET

    if dataset_name == 'mini-imagenet':
        datadir = os.path.join(config.DATASET.ROOT+'/images')
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
        
        test_dataset = miniimagenet(datadir, shots=1, ways=5, meta_test=False, transform=transform_valid, download=True)
      
    elif dataset_name == 'omniglot':
        train_val_permutation = list(range(1623))
        task_defs = [dataset.OmniglotTask(train_val_permutation, root=config.DATASET.ROOT, num_cls=config.IMAML.N_WAY, num_inst=config.IMAML.K_SHOT) for _ in range(config.IMAML.NUM_TEST_TASKS)]
        test_dataset = dataset.OmniglotFewShotDataset(task_defs=task_defs, img_size=config.MODEL.IMAGE_SIZE[1], GPU=False)

    elif dataset_name == 'sinusoid':
        test_dataset = dataset.SinusoidDataset(num_tasks=100000, GPU=False)

    sampler = torch.utils.data.RandomSampler(test_dataset, replacement=True, generator=torch.Generator(device='cuda'))
    if dataset_name == 'mini-imagenet':
        test_loader = BatchMetaDataLoader(test_dataset,
                             batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
                             shuffle=True,
                             num_workers=1)
    else:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
            sampler=sampler,
            shuffle=sampler is None,
            drop_last=True,
        )

    test(config, test_loader, model, criterion)

if __name__ == '__main__':
    main()
