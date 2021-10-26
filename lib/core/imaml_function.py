# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
# import ipdb
import torch
import sys
from utils.utils import save_checkpoint
import gc


logger = logging.getLogger(__name__)


def train(rank, config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
    output_dir, tb_log_dir, writer_dict, niter, args, visualizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    CE_losses = AverageMeter()
    jac_losses = AverageMeter()
    grad_norms = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    total_batch_num = len(train_loader)
    effec_batch_num = int(config.PERCENT * total_batch_num)
    writer = writer_dict['writer'] if (rank==0) else None
    writer_dict = writer_dict if (rank==0) else None

    for i, inputs in enumerate(train_loader):
        if config.DATASET.DATASET == 'mini-imagenet':
            x_train, y_train = inputs['train']
            x_val, y_val = inputs['test']
            x_train, y_train, x_val, y_val = x_train.cuda(), y_train.cuda(), x_val.cuda(), y_val.cuda().reshape(-1)
        else:
            x_train = inputs['x_train'].cuda()
            y_train = inputs['y_train'].cuda()
            x_val = inputs['x_val'].cuda()
            y_val = inputs['y_val'].cuda().reshape(-1)

        if i >= effec_batch_num:
            break
            
        # measure data loading time
        data_time.update(time.time() - end)

        y_val_hat, jac_loss = model(x_train, y_train, x_val, train_step=(niter[0]-1), writer=writer)
        
        if y_val.dtype == torch.float32:
            CE_loss = criterion(y_val_hat, y_val.to(y_val_hat.device).unsqueeze(-1))
        else:
            CE_loss = criterion(y_val_hat, y_val.to(y_val_hat.device))
        accuracy = (torch.argmax(y_val_hat, dim=1) == y_val.to(y_val_hat.device)).float().mean()
        jac_loss = jac_loss.mean()
        loss = CE_loss + jac_loss*config.TRAIN.JAC_COEFF

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        if config['TRAIN']['CLIP'] > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['TRAIN']['CLIP'])
            grad_norms.update(grad_norm.item(), y_val.size(0))
        
        optimizer.step()
        if config.TRAIN.LR_SCHEDULER != 'step':
            lr_scheduler.step()
            niter[0] +=1
        torch.cuda.empty_cache()
        
        # measure accuracy and record loss
        CE_losses.update(CE_loss.item(), y_val.size(0))
        jac_losses.update(jac_loss.item(), y_val.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        gc.collect()
        
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Memory {memory:.1f} \t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'CE Loss {CE_loss.val:.5f} ({CE_loss.avg:.5f})\t' \
                  'Accuracy {accuracy:.5f}\t' \
                  'Jac Loss {jac_loss.val:.5f} ({jac_loss.avg:.5f})\t' \
                  'Grad Norm {grad_norm.val:.5f} ({grad_norm.avg:.5f})\t' .format(
                      epoch, i, effec_batch_num, batch_time=batch_time,
                      speed=x_val.size(0)/batch_time.val,
                      data_time=data_time, CE_loss=CE_losses, jac_loss=jac_losses,
                      grad_norm=grad_norms, accuracy=accuracy, memory=torch.cuda.memory_allocated())
            logger.info(msg)

            if (rank==0) and writer_dict['writer']:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss_ce', CE_losses.val, global_steps)
                writer.add_scalar('train_loss_jac', jac_losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

            if (rank==0) and i%500==0:
                logger.info('=> saving checkpoint to {}'.format(output_dir))
                filename = 'checkpoint' + str(i) + '.pth.tar'
                save_checkpoint({
                    'iter' : i,
                    'epoch': epoch + 1,
                    'model': config.MODEL.NAME,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                }, False, output_dir, filename=filename)


def test(config, test_loader, model, criterion):
    # switch to eval mode
    model.eval()
    accuracies = []
    timings = []

    end = time.time()
    total_batch_num = len(test_loader)
    effec_batch_num = int(config.PERCENT * total_batch_num)
    writer = None

    for i, inputs in enumerate(test_loader):

        if config.DATASET.DATASET == 'mini-imagenet':
            x_train, y_train = inputs['train']
            x_val, y_val = inputs['test']
            x_train, y_train, x_val, y_val = x_train.cuda(), y_train.cuda(), x_val.cuda(), y_val.cuda().reshape(-1)
        else:
            x_train = inputs['x_train'].cuda()
            y_train = inputs['y_train'].cuda()
            x_val = inputs['x_val'].cuda()
            y_val = inputs['y_val'].cuda().reshape(-1)

        if i >= effec_batch_num:
            break

        with torch.no_grad():
            torch.cuda.synchronize(device=None)
            start = time.time()
            y_val_hat, jac_loss = model(x_train, y_train, x_val, train_step=-1, writer=writer, proj_adam=config.MODEL.PROJ_GD)
            torch.cuda.synchronize(device=None)
            end = time.time()
        timings.append(end - start)
        
        accuracy = (torch.argmax(y_val_hat, dim=1) == y_val.to(y_val_hat.device)).float().mean()
        accuracies.append(accuracy)
        print('iter : {}, time taken : {}, accuracy : {}'.format(i,end - start, accuracy))
        
        torch.cuda.empty_cache()
        
        
        gc.collect()

    print("Avg accuracy : {}, Avg time taken : {}".format(np.mean(accuracies), np.mean(timings[1:])))

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