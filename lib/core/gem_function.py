# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import torch
import sys
import numpy as np
from utils.utils import save_checkpoint
# import ipdb
import gc
logger = logging.getLogger(__name__)


def train_gem(config, train_loader, model, criterion, criterionsq, optimizer, lr_scheduler, epoch,
          output_dir, tb_log_dir, writer_dict, niter, losses_recent, loss_best, visualizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    rec_losses = AverageMeter()
    sq_losses = AverageMeter()
    jac_losses = AverageMeter()
    grad_norms = AverageMeter()
    # switch to train mode
    model.train()
    gc.collect()

    end = time.time()
    total_batch_num = len(train_loader)
    effec_batch_num = int(config.PERCENT * total_batch_num)
    for i, (input, _) in enumerate(train_loader):
        # train on partial training data
        if i >= effec_batch_num:
            break
            
        # measure data loading time
        data_time.update(time.time() - end)

        # perform JIIO and compute output
        rec, gx, ugrad, jac_loss = model(input, train_step=(niter[0]-1), writer=writer_dict['writer'])

        rec_loss = criterion(rec, input.to(rec.device))
        sq_loss = criterionsq(rec, input.to(rec.device))
        loss = rec_loss + jac_loss.mean()*config['TRAIN']['JAC_COEFF']
        losses_recent.append(rec_loss.item())
        if len(losses_recent)>80:
            losses_recent.pop(0)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        if config['TRAIN']['CLIP'] > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['TRAIN']['CLIP'])
            grad_norms.update(grad_norm.item(), input.size(0))
        optimizer.step()
        if config.TRAIN.LR_SCHEDULER != 'step':
            lr_scheduler.step()
            niter[0] +=1

        # record loss
        rec_losses.update(rec_loss.item(), input.size(0))
        sq_losses.update(sq_loss.item(), input.size(0))
        jac_losses.update(jac_loss.mean().item(), input.size(0))
        
        gc.collect()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Reconstruction Loss {rec_loss.val:.5f} ({rec_loss.avg:.5f})\t' \
                  'Squared Loss {sq_loss.val:.5f} ({sq_loss.avg:.5f})\t' \
                  'Jac Loss {jac_loss.val:.5f} ({jac_loss.avg:.5f})\t' \
                  'Grad Norm {grad_norm.val:.5f} ({grad_norm.avg:.5f})\t' .format(
                      epoch, i, effec_batch_num, batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, rec_loss=rec_losses, sq_loss=sq_losses,
                      jac_loss=jac_losses, grad_norm=grad_norms)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss_reconstruction', rec_losses.val, global_steps)
                writer.add_scalar('train_loss_jac', jac_losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

            if i%500==0:
                best = False
                if np.mean(losses_recent) < loss_best[0]:
                    loss_best[0] = np.mean(losses_recent)
                    best = True
                logger.info('=> saving checkpoint to {}'.format(output_dir))
                filename = 'checkpoint' + str(epoch) + str(i) + '.pth.tar'
                save_checkpoint({
                    'iter' : i,
                    'epoch': epoch + 1,
                    'model': config.MODEL.NAME,
                    'state_dict': model.module.state_dict(),
                    # 'perf': perf_indicator,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                }, best, output_dir, filename=filename)


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