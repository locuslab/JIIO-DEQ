# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ipdb
import time
import logging

import torch
import numpy as np
from core.cls_evaluate import accuracy


logger = logging.getLogger(__name__)

def print_memory():
        # meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
        # max_bytes_in_use = meminfo.used
        # memory = Memory(max_bytes_in_use)
        # print(memory)

        # print(torch.cuda.max_memory_allocated()/(1024*1024*1024))

        print(0, torch.cuda.max_memory_allocated(0)/(1024*1024*1024), torch.cuda.memory_reserved(0)/(1024*1024*1024), torch.cuda.memory_allocated(0)/(1024*1024*1024), torch.cuda.max_memory_reserved(0)/(1024*1024*1024))#, torch.cuda.memory_allocated(0)/(1024*1024*1024))
        # print(1, torch.cuda.max_memory_allocated(1)/(1024*1024*1024), torch.cuda.memory_reserved(1)/(1024*1024*1024), torch.cuda.memory_allocated(1)/(1024*1024*1024), torch.cuda.max_memory_reserved(1)/(1024*1024*1024))#, torch.cuda.memory_allocated(0)/(1024*1024*1024))
        # print(2, torch.cuda.max_memory_allocated(2)/(1024*1024*1024), torch.cuda.memory_reserved(2)/(1024*1024*1024), torch.cuda.memory_allocated(2)/(1024*1024*1024), torch.cuda.max_memory_reserved(2)/(1024*1024*1024))#, torch.cuda.memory_allocated(0)/(1024*1024*1024))

def train(config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
    output_dir, tb_log_dir, writer_dict, topk=(1,5)):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    total_batch_num = len(train_loader)
    effec_batch_num = int(config.PERCENT * total_batch_num)
    for i, (input, target) in enumerate(train_loader):
        # train on partial training data
        if i >= effec_batch_num:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        # print_memory()
        output, jac_loss = model(input, target, train_step=(lr_scheduler._step_count-1), writer=writer_dict['writer'])
        target = target.cuda(non_blocking=True)
        loss = criterion(output, target) + config['TRAIN']['JAC_COEFF'] * jac_loss.mean()
        # print_memory()
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        if config['TRAIN']['CLIP'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['TRAIN']['CLIP'])
        optimizer.step()
        if config.TRAIN.LR_SCHEDULER != 'step':
            lr_scheduler.step()
        
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        prec1, prec5 = accuracy(output, target, topk=topk)
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print_memory()
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                  'Accuracy@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                      epoch, i, effec_batch_num, batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, top1=top1, top5=top5)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_top1', top1.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1


def validate(config, val_loader, model, criterion, lr_scheduler, epoch, output_dir, tb_log_dir,
    writer_dict=None, topk=(1,5)):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    jac_loss = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            output, jl = model(input, target,
                   train_step=-1,
                   writer=writer_dict['writer'] if writer_dict is not None else None,)
            
            target = target.cuda(non_blocking=True)
            loss = criterion(output, target)
            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            prec1, prec5 = accuracy(output, target, topk=topk)
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
        msg = 'Test: Time {batch_time.avg:.3f}\t' \
              'Loss {loss.avg:.4f}\t' \
              'Error@1 {error1:.3f}\t' \
              'Error@5 {error5:.3f}\t' \
              'Accuracy@1 {top1.avg:.3f}\t' \
              'Accuracy@5 {top5.avg:.3f}\t'.format(
                  batch_time=batch_time, loss=losses, top1=top1, top5=top5,
                  error1=100-top1.avg, error5=100-top5.avg)
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_top1', top1.avg, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return top1.avg

def validate_proj(config, val_loader, model, criterion, lr_scheduler, epoch, output_dir, tb_log_dir,
    writer_dict=None, topk=(1,5)):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # compute output
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        interval = []
        torch.cuda.synchronize(device=None)
        start = time.time()
        delta = torch.nn.parameter.Parameter(torch.zeros_like(input))
        # initialize optimizer
        epsilon = 1
        optimizer = torch.optim.SGD([delta,],
                               lr=0.2,)
        for j in range(20):
            torch.cuda.synchronize(device=None)
            start1 = time.time()
            output, jl = model(input+delta,
                           train_step=-1,      
                           writer=writer_dict['writer'] if writer_dict is not None else None,)
            interval.append(time.time() - start1)
                           
            target = target.cuda(non_blocking=True)
            loss = -criterion(output, target)
            delta_grad = torch.autograd.grad(loss, delta)[0]
            delta.grad = delta_grad.detach()/(delta_grad.data.norm(dim=(2,3), keepdim=True).norm(dim=(1), keepdim=True)+1e-8)
                
            optimizer.step()
            optimizer.zero_grad()

            delta_norms = delta.data.norm(dim=(2,3), keepdim=True).norm(dim=(1), keepdim=True)+1e-8
            factor = epsilon / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta.data = delta.data * factor.view(-1, 1,1,1)

        torch.cuda.synchronize(device=None)
        end = time.time() - start
        
        loss = criterion(output, target)
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        prec1, prec5 = accuracy(output, target, topk=topk)
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
    msg = 'Test: Time {batch_time.avg:.3f}\t' \
          'Loss {loss.avg:.4f}\t' \
          'Error@1 {error1:.3f}\t' \
          'Error@5 {error5:.3f}\t' \
          'Accuracy@1 {top1.avg:.3f}\t' \
          'Accuracy@5 {top5.avg:.3f}\t'.format(
              batch_time=batch_time, loss=losses, top1=top1, top5=top5,
              error1=100-top1.avg, error5=100-top5.avg)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_top1', top1.avg, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return top1.avg

def train_proj(config, train_loader, model, criterion, optimizer, lr_scheduler, epoch, output_dir, tb_log_dir,
    writer_dict=None, topk=(1,5)):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.train()

    end = time.time()
    total_batch_num = len(train_loader)
    effec_batch_num = int(config.PERCENT * total_batch_num)
    for i, (input, target) in enumerate(train_loader):
        # compute output
        # print_memory()
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        delta = torch.nn.parameter.Parameter(torch.zeros_like(input))
        # initialize optimizer
        epsilon = 1
        optimizer_inner = torch.optim.SGD([delta,],
                               lr=0.2,)
        for j in range(20):
            output, jac_loss = model(input+delta,
                           train_step=-1,       
                           writer=None,)
                           
            loss = -criterion(output, target)
            delta_grad = torch.autograd.grad(loss, delta)[0]
            delta.grad = delta_grad.detach()/(delta_grad.data.norm(dim=(2,3), keepdim=True).norm(dim=(1), keepdim=True)+1e-8)
                
            optimizer_inner.step()
            optimizer_inner.zero_grad()

            delta_norms = delta.data.norm(dim=(2,3), keepdim=True).norm(dim=(1), keepdim=True)+1e-8
            factor = epsilon / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta.data = delta.data * factor.view(-1, 1,1,1)

        output, jac_loss = model(input+delta.detach(), target,
                       train_step=(lr_scheduler._step_count-1),       # Evaluate using MDEQ (even when pre-training)
                       writer=writer_dict['writer'] if writer_dict is not None else None, proj_adam=True)
        loss = criterion(output, target) + config['TRAIN']['JAC_COEFF'] * jac_loss.mean()
        # print_memory()
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        if config['TRAIN']['CLIP'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['TRAIN']['CLIP'])
        optimizer.step()
        if config.TRAIN.LR_SCHEDULER != 'step':
            lr_scheduler.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        prec1, prec5 = accuracy(output, target, topk=topk)
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print_memory()
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                  'Accuracy@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                      epoch, i, effec_batch_num, batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, top1=top1, top5=top5)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_top1', top1.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1



def timing_experiments(config, val_loader, model, criterion, lr_scheduler, epoch, output_dir, tb_log_dir,
    writer_dict=None, topk=(1,5)):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    intervals_1 = []
    intervals_2 = []
    intervals_3 = []
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        model.eval()
        jac_loss = []
        start2 = torch.cuda.Event(enable_timing=True)
        end2 = torch.cuda.Event(enable_timing=True)
        start2.record()

        with torch.no_grad():
            
            output, jl = model(input, target,
                   train_step=-1,       
                   writer=writer_dict['writer'] if writer_dict is not None else None,)
        end2.record()
        torch.cuda.synchronize(device=None)
        # intervals_1.append(model.module.deq.ugrad.cpu().numpy())
        intervals_3.append(start2.elapsed_time(end2))
        # print(start2.elapsed_time(end2), (output.argmax(dim=1) == target).float().mean()*100)
        model.train()
        delta = torch.nn.parameter.Parameter(torch.zeros_like(input))
        # initialize optimizer
        interval = []
        start2.record()
        epsilon = 1
        optimizer = torch.optim.SGD([delta,],
                               lr=0.6,)
        for j in range(20):
            output, jl = model(input+delta,
                           train_step=-1,       # Evaluate using MDEQ (even when pre-training)
                           writer=writer_dict['writer'] if writer_dict is not None else None,)
            
            target = target.cuda(non_blocking=True)
            loss = -criterion(output, target)
            delta_grad = torch.autograd.grad(loss, delta)[0]
            delta.grad = delta_grad.detach()/(delta_grad.data.norm(dim=(2,3), keepdim=True).norm(dim=(1), keepdim=True)+1e-8)
                
            optimizer.step()
            optimizer.zero_grad()

            delta_norms = delta.data.norm(dim=(2,3), keepdim=True).norm(dim=(1), keepdim=True)+1e-8
            factor = epsilon / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta.data = delta.data * factor.view(-1, 1,1,1)
            end2.record()
            torch.cuda.synchronize(device=None)
            interval.append([j, start2.elapsed_time(end2), loss])

        if interval[-1][1] <3000:
            continue
        intervals_2.append(np.array(interval))
        print(i, interval[-1][1], intervals_3[-1], (output.argmax(dim=1) == target).float().mean()*100)
        ipdb.set_trace()

        # if i%20==0 and i>0:
        #     print('finished', i)
        #     save_folder = '/'.join(config.TEST.MODEL_FILE.split('/')[:-1])
        #     np.save(save_folder + '/intervals_1.npy', intervals_1)
        #     np.save(save_folder + '/intervals_2.npy', intervals_2)
        #     np.save(save_folder + '/intervals_3.npy', intervals_3)
        #     break



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
