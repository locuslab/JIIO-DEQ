# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import sys
# import pdb

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.nn import functional as F
from tensorboardX import SummaryWriter

import _init_paths
import models
from config import config
from config import update_config
from core.vae_function import train
from utils.modelsummary import get_model_summary
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import DensityEstimator
from utils.utils import fit_density_model
from termcolor import colored
from natsort import natsorted
from PIL import Image
import numpy as np
import time
# from datasets.ffhq import FFHQ_Dataset
from torchvision.utils import save_image
# import ipdb
import gc

import py3nvml.py3nvml as nvml
nvml.nvmlInit()
handle = nvml.nvmlDeviceGetHandleByIndex(0)
from typing import Callable, Iterable, List, NamedTuple, Optional, Union

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
    parser.add_argument('--task',
                        help='Task label',
                        default='timing')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()
    seed = 1265
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    print(colored("Setting default tensor type to cuda.FloatTensor", "cyan"))
    torch.multiprocessing.set_start_method('spawn')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    model = eval('models.'+config.MODEL.NAME+'.get_jiio_vae')(config).cuda()

    
    if config.TEST.MODEL_FILE:
        try:
            model.load_state_dict(torch.load(config.TEST.MODEL_FILE))#, map_location=torch.device('cpu')))
        except:
            model.load_state_dict(torch.load(config.TEST.MODEL_FILE)['state_dict'])
 
    ### Uncomment from here all the way to the end
    gpus = list(config.GPUS)
    num_gpus=1
    if args.task == 'psnr' or 'get_samples_for_fid':
        model = nn.DataParallel(model, device_ids=gpus).cuda()
        num_gpus = len(gpus)
    print("Finished constructing model!", gpus)

    criterion = lambda output, target: ((output - target)**2).mean()

    optimizer = get_optimizer(config, model)
    lr_scheduler = None

    best_perf = 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH

    dataset_name = config.DATASET.DATASET

    if dataset_name == 'imagenet':
        traindir = os.path.join(config.DATASET.ROOT+'/images', config.DATASET.TRAIN_SET)
        valdir = os.path.join(config.DATASET.ROOT+'/images', config.DATASET.TEST_SET)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.ImageFolder(traindir, transform_train)
        test_dataset = datasets.ImageFolder(testdir, transform_test)

    elif dataset_name == 'celeba':
        class CelebADataset(Dataset):
          def __init__(self, root_dir, transform=None, test=False):
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
            if test:
                self.images = self.test_names
            else:
                self.images = self.train_names

          def __len__(self): 
            return len(self.images)

          def save_test_set(self):
            for img_name in self.test_names:
                img_path = os.path.join(self.root_dir, img_name)
                img = Image.open(img_path)
                new_img = self.transform(img)
                png_filename = f'{config.DATASET.ROOT}/celeba64/'+img_name
                new_img.save(png_filename)

          def __getitem__(self, idx):
            # Get the path to the image 
            img_path = os.path.join(self.root_dir, self.images[idx])
            # Load image and convert it to RGB
            img = Image.open(img_path).convert('RGB')
            # Apply transformations to the image
            if self.transform:
              img = self.transform(img)
            # print(idx)
            return img, img[0, 0, 0]

        ## Load the dataset 
        # Path to directory with all the images
        img_folder = f'{config.DATASET.ROOT}/celeba/img_align_celeba'

        # Load the dataset from file and apply transformations
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        if config.MODEL.OUTPUT_NL == 'sigmoid':
            transform_test = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(config.MODEL.IMAGE_SIZE[1]),
                                            transforms.ToTensor()
                                            ])
        else:
            transform_test = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(config.MODEL.IMAGE_SIZE[1]),
                                            transforms.ToTensor(),
                                            normalize
                                            ])

        test_dataset = CelebADataset(img_folder, transform_test, test=True)
        train_dataset = CelebADataset(img_folder, transform_test)

    elif dataset_name == "cifar10":
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # For reference
        
        # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        augment_list = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if config.DATASET.AUGMENT else []
        if config.MODEL.OUTPUT_NL == 'sigmoid':
            transform_test = transforms.Compose(augment_list + [
                transforms.ToTensor(),
            ])
        else:
            transform_test = transforms.Compose(augment_list + [
                transforms.ToTensor(),
                normalize,
            ])
        test_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=True, download=True, transform=transform_train)
    elif dataset_name == "mnist":
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        test_dataset = datasets.MNIST(root=f'{config.DATASET.ROOT}', train=True, download=True, transform=transform_train)

    elif dataset_name == "ffhq":
        test_dataset = FFHQ_Dataset(config.DATASET.ROOT, config.MODEL.IMAGE_SIZE[-1], transparent = False, aug_prob = 0)
           
    if dataset_name == "celeba":
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*num_gpus,
            # shuffle=True,
            # num_workers=config.WORKERS,
            drop_last=True,
            pin_memory=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*num_gpus,
            # shuffle=True,
            # num_workers=config.WORKERS,
            drop_last=True,
            pin_memory=True
        )
    else:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*num_gpus,
            # shuffle=True,
            num_workers=config.WORKERS,
            pin_memory=True
        )

    def save_images_for_fid(save_dir, latents=None):
        model.eval()
        try:
            os.system('mkdir ' + save_dir)
        except:
            pass
        for i, (input, _) in enumerate(test_loader):
            input = input.cuda()
            if latents is not None:
                latent = torch.tensor(latents[i*input.shape[0]:(i+1)*input.shape[0]]).cuda()
                rec = model(x=None, train_step=1e10, latent=latent, writer=None)
            else:
                rec, _, _, _ = model(x, train_step=1e10, writer=None)

            for j in range(rec.shape[0]):
                save_image((rec[j]+1)/2, save_dir+str(i*rec.shape[0]+j)+'.jpg')

    def bytes_to_mega_bytes(memory_amount: int) -> int:
        """Utility to convert a number of bytes (int) into a number of mega bytes (int)"""
        return memory_amount >> 20
    
    def print_memory():
        print(torch.cuda.max_memory_allocated()/(1024*1024*1024), torch.cuda.memory_reserved()/(1024*1024*1024), torch.cuda.memory_allocated()/(1024*1024*1024), torch.cuda.max_memory_reserved()/(1024*1024*1024))

    def timing_results():
        model.train()
        latent_shape = 128 
        for i, (input, _) in enumerate(train_loader):
            bsz = 1
            input = input[:bsz].cuda()
            start2 = torch.cuda.Event(enable_timing=True)
            end2 = torch.cuda.Event(enable_timing=True)
            latent = model(input, train_step=1e10, writer=None, get_latent=True)
            start2.record()
            output, _, _, _ = model(input, train_step=1e10, writer=None)
            end2.record()
            torch.cuda.synchronize(device=None)
            print('JIIO : ', start2.elapsed_time(end2))

            start1 = torch.cuda.Event(enable_timing=True)
            end1 = torch.cuda.Event(enable_timing=True)

            z = torch.nn.parameter.Parameter(torch.zeros((input.shape[0], latent_shape))).to(input)
            # initialize optimizer
            optimizer = optim.Adam([z,],
                                   lr=1,)
            targ = input
            imgsize = input.shape
            interval = []
            x = model._decode(z, -1)
            start1.record()
            # compute objective
            optimizer.zero_grad()
            x = model._decode(z, -1)

            # loop over the iterations
            for j in range(50):

                # compute gradients
                cost = F.mse_loss(x, targ)
                cost1 = F.mse_loss(x, input)

                # compute updates
                z_grad = torch.autograd.grad(cost, [z])[0]
                z.grad = z_grad.detach()

                # optimize the latents
                optimizer.step()
                end1.record()
                torch.cuda.synchronize()
                interval.append(np.array([j, start1.elapsed_time(end1), cost.item(), cost1.item()]))
                
                # compute objective
                optimizer.zero_grad()
                x = model._decode(z, -1)

            # intervals_2.append(np.array(interval))
            end1.record()
            torch.cuda.synchronize()
            print('PGD : ', start1.elapsed_time(end1))

    def get_psnr_test(opt='jiio'):
        model.train()
        losses = []
        latent_shape = 128 
        for i, (input, _) in enumerate(test_loader):
            input = input.cuda()
            if opt=='jiio':
                output, _, _, _ = model(input, train_step=1e10, writer=None)
                cost = F.mse_loss(output, input)
                losses.append(cost.item())
                # ipdb.set_trace()
            else:
                z = torch.nn.parameter.Parameter(torch.zeros((input.shape[0], latent_shape))).to(input)
                # initialize optimizer
                optimizer = optim.Adam([z,],
                                       lr=1,)
                targ = input
                imgsize = input.shape
                x = model._decode(z, -1)
                start1.record()
                optimizer.zero_grad()
                x = model._decode(z, -1)

                # loop over the iterations
                for j in range(40):
                    cost = F.mse_loss(x, targ)
                    cost1 = F.mse_loss(x, input)

                    # compute updates
                    z_grad = torch.autograd.grad(cost, [z])[0]
                    z.grad = z_grad.detach()

                    # optimize the latents
                    optimizer.step()
                    optimizer.zero_grad()

                    x = model._decode(z, -1)
                cost = F.mse_loss(x, input)
                losses.append(cost.item())

        loss_array = np.asarray(losses)
        loss_mse = np.mean(loss_array)
        PSNR = -10 * np.log10(loss_mse)
        percentiles = np.percentile(loss_array, [25,50,75])
        percentiles = -10.0*np.log10(percentiles)
        print("TEST LOSS: " + str(sum(losses) / len(losses)), flush=True)
        print("MEAN TEST PSNR: " + str(PSNR), flush=True)
        print("TEST PSNR QUARTILES AND MEDIAN: " + str(percentiles[0]) +
              ", " + str(percentiles[1]) + ", " + str(percentiles[2]), flush=True)


    def save_test_set_for_fid():
        test_dataset.save_test_set()

    def latent_density_fit_and_save_samples():
        model.eval()
        latents = []
        with torch.no_grad():
            for i, (input, _) in enumerate(train_loader):
                input = input.cuda()
                latent = model(input, train_step=1e10, writer=None, get_latent=True)
                latents.append(latent.detach().clone().cpu())
                gc.collect()
                if (i+1)%20==0:
                    print('iterations :', i)
        latents = torch.cat(latents, dim=0)
        samples, sampler = fit_density_model(latents=latents.numpy())

        save_images_for_fid('./model_samples/', samples)



    if args.task=='timing':
        timing_results()
    elif args.task=='psnr':
        get_psnr_test()
    elif args.task=='get_samples_for_fid':
        latent_density_fit_and_save_samples()
    elif args.task=='save_test_set':
        save_test_set_for_fid()

if __name__ == '__main__':
    main()