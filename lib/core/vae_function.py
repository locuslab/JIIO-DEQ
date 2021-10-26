# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import numpy as np
import torch
from utils.utils import save_checkpoint

logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, criterionsq, optimizer, lr_scheduler, epoch,
          output_dir, tb_log_dir, writer_dict, niter, losses_recent, loss_best, visualizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    rec_losses = AverageMeter()
    kld_losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    total_batch_num = len(train_loader)
    effec_batch_num = int(config.PERCENT * total_batch_num)
    for i, (input, _) in enumerate(train_loader):
        # train on partial training data
        if i >= effec_batch_num:
            break
            
        # measure data loading time
        data_time.update(time.time() - end)

        imgsize = input.shape
        bsz = imgsize[0]
        
        mask = torch.ones(imgsize).to(input)
        if config.TRAIN.INV_PROB=='inpainting':
            center = np.random.randint(imgsize[-1]-20, size=2*bsz)+10
            for i in range(bsz):
                mask[i, :, center[2*i]-10:center[2*i]+10, center[2*i+1]-10:center[2*i+1]+10]*=0
        elif config.TRAIN.INV_PROB=='denoising02':
            targ = torch.normal(mean=torch.zeros_like(input), std=torch.zeros_like(input) + 0.2)
        elif config.TRAIN.INV_PROB=='denoising04':
            targ = torch.normal(mean=torch.zeros_like(input), std=torch.zeros_like(input) + 0.4)
        else:
            targ = torch.zeros_like(input)

        # compute output
        kld_loss, rec = model(input*mask+targ, train_step=(lr_scheduler._step_count-1), writer=writer_dict['writer'], vae_fwd=True)

        rec_loss = criterion(rec, input.to(rec.device))

        kld_loss = kld_loss.mean()/np.prod(input.shape[1:])
        loss = rec_loss + min(epoch / 10, 1) * kld_loss
        losses_recent.append(rec_loss.item())
        if len(losses_recent)>80:
            losses_recent.pop(0)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        if config['TRAIN']['CLIP'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['TRAIN']['CLIP'])
        optimizer.step()
        if config.TRAIN.LR_SCHEDULER != 'step':
            lr_scheduler.step()

        # measure accuracy and record loss
        rec_losses.update(rec_loss.item(), input.size(0))
        kld_losses.update(kld_loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:

            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Reconstruction Loss {rec_loss.val:.5f} ({rec_loss.avg:.5f})\t' \
                  'KLD Loss {kld_loss.val:.5f} ({kld_loss.avg:.5f})\t' .format(
                      epoch, i, effec_batch_num, batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, rec_loss=rec_losses, kld_loss=kld_losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss_reconstruction', rec_losses.val, global_steps)
                writer.add_scalar('train_loss_kld', kld_losses.val, global_steps)
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

def pre_process(x, num_bits):
    if num_bits != 8:
        x = torch.floor(x * 255 / 2 ** (8 - num_bits))
        x /= (2 ** num_bits - 1)
    return x