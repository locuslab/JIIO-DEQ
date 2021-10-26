# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import sys

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
from core.vae_function import train
from core.gem_function import train_gem
from utils.modelsummary import get_model_summary
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from termcolor import colored
from natsort import natsorted
from PIL import Image
import visdom


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
    args = parse_args()
    print(colored("Setting default tensor type to cuda.FloatTensor", "cyan"))
    torch.multiprocessing.set_start_method('spawn')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    viz=None

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    
    model = eval('models.'+config.MODEL.NAME+'.get_jiio_vae')(config).cuda()

    dump_input = torch.rand(config.TRAIN.BATCH_SIZE_PER_GPU, config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[2]).cuda()
    logger.info(get_model_summary(model, dump_input))
    
    if config.TRAIN.MODEL_FILE:
        model.load_state_dict(torch.load(config.TRAIN.MODEL_FILE))
        logger.info(colored('=> loading model from {}'.format(config.TRAIN.MODEL_FILE), 'red'))

    # copy model file
    this_dir = os.path.dirname(__file__)
    models_dst_dir = os.path.join(final_output_dir, 'models')
    if os.path.exists(models_dst_dir):
        shutil.rmtree(models_dst_dir)
    shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)
    models_dst_dir = os.path.join(final_output_dir, 'lib')
    if os.path.exists(models_dst_dir):
        shutil.rmtree(models_dst_dir)
    shutil.copytree(os.path.join(this_dir, '../lib'), models_dst_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    print("Finished constructing model!")

    # cross entropy
    if config.MODEL.OUTPUT_NL == 'sigmoid':
        criterion = lambda output, target: (- target * torch.log(output) - (1 - target) * torch.log(1 - output)).sum() / output.size(0) # cross entropy
    elif config.MODEL.OUTPUT_NL == 'tanh':
        criterion = lambda output, target: ((output - target)**2).mean()
    else:
        criterion = lambda output, target: ((output - target)**2).mean()
    criterionsq = lambda output, target: ((output - target)**2).mean()
    # criterion = lambda output, target: torch.abs(output - target).mean() # l1

    optimizer = get_optimizer(config, model)
    lr_scheduler = None

    best_perf = 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint1500.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            # best_perf = checkpoint['perf']
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

    if dataset_name == 'imagenet':
        traindir = os.path.join(config.DATASET.ROOT+'/train')
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transform_train = transforms.Compose([
            transforms.Resize(config.MODEL.IMAGE_SIZE[1]),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[1]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.ImageFolder(traindir, transform_train)

    elif dataset_name == 'celeba':
        class CelebADataset(Dataset):
          def __init__(self, root_dir, transform=None):
            """
            Args:
              root_dir (string): Directory with all the images
              transform (callable, optional): transform to be applied to each image sample
            """
            # Read names of images in the root directory
            image_names = os.listdir(root_dir)

            self.root_dir = root_dir
            self.transform = transform 
            self.image_names = natsorted(image_names)
            self.train_names = self.image_names[:162770]
            self.valid_names = self.image_names[162770:162770+19867]
            self.test_names = self.image_names[162770+19867:]

          def __len__(self): 
            return len(self.train_names)

          def __getitem__(self, idx):
            # Get the path to the image 
            img_path = os.path.join(self.root_dir, self.train_names[idx])
            # Load image and convert it to RGB
            img = Image.open(img_path).convert('RGB')
            # Apply transformations to the image
            if self.transform:
              img = self.transform(img)

            return img, img[0, 0, 0]

        # Path to directory with all the images
        img_folder = f'{config.DATASET.ROOT}/celeba/img_align_celeba'

        # Load the dataset from file and apply transformations 
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        if config.MODEL.OUTPUT_NL == 'sigmoid':
            transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(config.MODEL.IMAGE_SIZE[1]),
                                            transforms.ToTensor()])
        else:
            transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(config.MODEL.IMAGE_SIZE[1]),
                                            transforms.ToTensor(),
                                            normalize])

        train_dataset = CelebADataset(img_folder, transform_train)

    elif dataset_name == "cifar10":
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # For reference
        
        # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        augment_list = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if config.DATASET.AUGMENT else []
        if config.MODEL.OUTPUT_NL == 'sigmoid':
            transform_train = transforms.Compose(augment_list + [
                transforms.ToTensor(),
            ])
        else:
            transform_train = transforms.Compose(augment_list + [
                transforms.ToTensor(),
                normalize,
            ])
        train_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=True, download=True, transform=transform_train)
        
    elif dataset_name == "mnist":
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_dataset = datasets.MNIST(root=f'{config.DATASET.ROOT}', train=True, download=True, transform=transform_train)
    
    if dataset_name == "celeba":
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
            shuffle=True,
            num_workers=config.WORKERS,
            pin_memory=True
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
    losses_recent = []
    loss_best = [1e9]
    # Training code
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        if config.TRAIN.LR_SCHEDULER == 'step':
            lr_scheduler.step()
            niter[0] +=1
        
        # train for one epoch
        if config.TRAIN.TRAIN_VAE:
            train(config, train_loader, model, criterion, criterionsq, optimizer, lr_scheduler, epoch,
                  final_output_dir, tb_log_dir, writer_dict, niter, losses_recent, loss_best, visualizer=viz)
        else:
            train_gem(config, train_loader, model, criterion, criterionsq, optimizer, lr_scheduler, epoch,
                  final_output_dir, tb_log_dir, writer_dict, niter, losses_recent, loss_best, visualizer=viz)
        torch.cuda.empty_cache()
        
        best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': config.MODEL.NAME,
            'state_dict': model.module.state_dict(),
            # 'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }, best_model, final_output_dir, filename='checkpoint.pth.tar')

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
