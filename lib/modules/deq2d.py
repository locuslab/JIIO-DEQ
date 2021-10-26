# Modified based on the DEQ repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function, Variable
import torch.autograd as autograd
import numpy as np
import pickle
import sys
import os
from scipy.optimize import root
import time
from termcolor import colored
import copy
sys.path.append("../")
from modules.broyden import broyden, analyze_broyden, anderson, anderson_deq
from tqdm import tqdm
from scipy.optimize import minimize
# import ipdb
from tensorboardX import SummaryWriter
import gc
import logging
logger = logging.getLogger(__name__)


def norm_diff(new, old, show_list=False):
    if show_list:
        return [(new[i] - old[i]).norm().item() for i in range(len(new))]
    return np.sqrt(sum((new[i] - old[i]).norm().item()**2 for i in range(len(new))))


class DEQFunc2d(Function):
    """ Generic DEQ-JIIO module that uses Broyden/Anderson's method to find the DEQ/JIIO/PGD equilibrium state """

    @staticmethod
    def f(func, z1, u, *args):
        return func(z1, u, v=None)

    @staticmethod
    def g(func, z1, u, cutoffs, *args):
        z1_list = DEQFunc2d.vec2list(z1, cutoffs)
        return DEQFunc2d.list2vec(DEQFunc2d.f(func, z1_list, u, *args)) - z1

    @staticmethod
    def list2vec(z1_list):
        bsz = z1_list[0].size(0)
        return torch.cat([elem.reshape(bsz, -1, 1) for elem in z1_list], dim=1)

    @staticmethod
    def vec2list(z1, cutoffs):
        bsz = z1.shape[0]
        z1_list = []
        start_idx, end_idx = 0, cutoffs[0][0] * cutoffs[0][1] * cutoffs[0][2]
        for i in range(len(cutoffs)):
            z1_list.append(z1[:, start_idx:end_idx].view(bsz, *cutoffs[i]))
            if i < len(cutoffs)-1:
                start_idx = end_idx
                end_idx += cutoffs[i + 1][0] * cutoffs[i + 1][1] * cutoffs[i + 1][2]
        return z1_list

    @staticmethod
    def broyden_find_root(func, z1, u, eps, *args):
        """
        Computing root for f(z1,u) - z1 using broyden's method 
        Note: u remains fixed.
        """
        bsz = z1[0].size(0)
        z1_est = DEQFunc2d.list2vec(z1)
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z1]
        threshold, train_step, writer = args[-3:]

        g = lambda x: DEQFunc2d.g(func, x, u, cutoffs, *args)
        result_info = broyden(g, z1_est, threshold=threshold, eps=eps, name="forward")
        z1_est = result_info['result']
        nstep = result_info['nstep']
        lowest_step = result_info['lowest_step']
        diff = result_info['diff']
        r_diff = min(result_info['new_trace'][1:])
        if z1_est.get_device() == 0:
            if writer is not None:
                writer.add_scalar('forward/diff', result_info['diff'], train_step)
                writer.add_scalar('forward/nstep', result_info['nstep'], train_step)
                writer.add_scalar('forward/lowest_step', result_info['lowest_step'], train_step)
                writer.add_scalar('forward/final_trace', result_info['new_trace'][lowest_step], train_step)

        status = analyze_broyden(result_info, judge=True)
        if status:
            err = {"z1": z1}
            analyze_broyden(result_info, err=err, judge=False, name="forward", save_err=False)
        
        if threshold > 30:
            torch.cuda.empty_cache()
        return z1_est.clone().detach()

    @staticmethod
    def anderson_find_root(func, z1, u, eps, *args):
        """
        Computing fixed point for z1 := f(z1,u) using anderson acceleration. 
        Note: u remains fixed.
        """
        bsz = z1[0].size(0)
        z1_est = DEQFunc2d.list2vec(z1)
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z1]
        threshold, train_step, writer = args[-3:]

        f = lambda x: DEQFunc2d.g(func, x, u, cutoffs, *args) + x
        result_info = anderson_deq(f, z1_est, threshold=threshold, eps=eps, beta=1.0, name="forward")
        z1_est = result_info['result']

        if threshold > 30:
            torch.cuda.empty_cache()
        return z1_est.clone().detach()

    @staticmethod
    def anderson_gem_find_root(func, z1, u, eps, *args):
        """
        Performing JIIO updates to otain fixed point updates on [z,u,mu] := F([z,u,mu],y) using anderson
        Note: both z1 and u are being optimized.
        """
        bsz = z1[0].size(0)
        z1_est = DEQFunc2d.list2vec(z1)
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z1]
        adv_tr, epsilon, mem, alpha_0, threshold, train_step, writer = args[-7:]
        z_size, x_size = z1_est.shape[1], u.shape[1]

        beta = 1.0
        eps = 1e-5
        lam = 1e-4
        alpha = [alpha_0[0],0.6, 0.8]
        iter_switch = max(50, int(threshold*5/8))

        interval = []
        start = time.time()
        start2 = torch.cuda.Event(enable_timing=True)
        end2 = torch.cuda.Event(enable_timing=True)
        start2.record()

        jc_u = []
        jc_z = []
        indices = np.random.randint(low=15, high=threshold, size=z1_est.shape[0])

        def f(v, it=0, proj=False):
            """
            Computes updates [z,u,mu] := F([z,u,mu],y)
            """
            v = v.squeeze(-1)
            z, x, mu = v[:, :z_size], v[:, z_size:z_size+x_size], v[:, z_size+x_size:2*z_size+x_size]
            if proj:
                # projection to norm ball
                x_temp = x
                delta_norms = x_temp.norm(dim=1, keepdim=True)+1e-8
                factor = epsilon / delta_norms
                factor = torch.min(factor, torch.ones_like(delta_norms))
                x_temp = x_temp * factor.view(-1, 1)
                return torch.cat([z, x_temp, mu], dim=1)
            with torch.enable_grad():
                z = Variable(z, requires_grad = True).to(z.device)
                x = Variable(x, requires_grad = True).to(x.device)
                if adv_tr:
                    z_out, recon_x, _, recon_x_vec, _ = func(torch.cat([z, x+u], dim=1), 0, 0, only_fx_cost=True)
                else:
                    z_out, recon_x, _, _, recon_x_vec = func(torch.cat([z, x], dim=1), 0, 0, only_fx_cost=True)
                mu_out, x_out = autograd.grad([recon_x, z_out], [z, x], grad_outputs=[None, mu])

                # Store certain iterates to perform jacobian regularization
                update_idx = indices==it
                jc_z.append(z[update_idx])
                if adv_tr:
                    jc_u.append((x+u)[update_idx])
                else:    
                    jc_u.append(x[update_idx])

                if (it+1)%iter_switch==0: 
                    alpha[0] = alpha_0[1]

                mu_temp = mu + alpha[1] * (mu_out - mu)
                z_temp = z + alpha[2] * (z_out - z)
                if adv_tr:
                    # projection to norm ball
                    x_temp = (x - alpha[0] * x_out/(x_out.norm(dim=1, keepdim=True)+1e-8))
                    delta_norms = x_temp.norm(dim=1, keepdim=True)+1e-8
                    factor = epsilon / delta_norms
                    factor = torch.min(factor, torch.ones_like(delta_norms))
                    x_temp = x_temp * factor.view(-1, 1)
                else:
                    x_temp = x - alpha[0] * x_out

                end2.record()
                torch.cuda.synchronize()
                interval.append([it, start2.elapsed_time(end2), recon_x.item(), (z_out - z).norm().item()])
            return torch.cat([z_temp, x_temp, mu_temp], dim=1).unsqueeze(-1), recon_x_vec
        g = lambda v, it: f(v, it) - v

        # Perform anderson acceleration to compute fixed points on F
        if adv_tr:
            result_info = anderson(f, torch.cat([z1_est, torch.zeros_like(u).unsqueeze(-1), z1_est*0], dim=1), z_size, m=mem, lam=lam, threshold=threshold, eps=eps, beta=beta)
            u_est = result_info['result'][:, z_size:z_size + x_size] + u.unsqueeze(-1)
        else:
            result_info = anderson(f, torch.cat([z1_est, u.unsqueeze(-1), z1_est*0], dim=1), z_size, m=mem, lam=lam, threshold=threshold, eps=eps, beta=beta)
            u_est = result_info['result'][:, z_size:z_size + x_size]
        gc.collect()
        z1_est = result_info['result'][:, :z_size]
        mu_final = result_info['result'][:, z_size + x_size:2*z_size + x_size]

        jc_z = torch.cat(jc_z, dim=0)
        jc_u = torch.cat(jc_u, dim=0)

        if z1_est.get_device() == 0:
            if writer is not None:
                writer.add_scalar('forward/diff', result_info['gx'][:, :z_size].norm().item(), train_step)
                writer.add_scalar('forward/lowest_step', result_info['nstep'].mean(), train_step)
                writer.add_scalar('forward/diff_normalized', result_info['gx'][:, :z_size].norm()/z1_est.norm(), train_step)
                writer.add_scalar('forward/mu_norm', result_info['result'][:, -z_size:].norm().item(), train_step)
                writer.add_scalar('forward/gxz_norm', result_info['gx'][:, z_size:z_size + x_size].norm().item(), train_step)
                writer.add_scalar('forward/gxmu_norm', result_info['gx'][:, -z_size:].norm().item(), train_step)
                zmean = result_info['result'][:, z_size:z_size + x_size].mean(dim=0, keepdim=True)
                zvar = ((result_info['result'][:, z_size:z_size + x_size] - zmean)**2).mean(dim=0, keepdim=True)
                writer.add_scalar('forward/zmean', zmean.norm().item(), train_step)
                writer.add_scalar('forward/zvar', zvar.norm().item(), train_step)
                
        return z1_est.squeeze(-1), u_est.squeeze(-1), mu_final.squeeze(-1), jc_z, jc_u
        
    @staticmethod
    def broyden_projected_adam(func, z1, u, eps, *args):
        """
        Performing Projected Adam updates on inputs u. The projection here is on the constraints z = f(z,u) using broyden
        """
        bsz = z1[0].size(0)
        z1_est = DEQFunc2d.list2vec(z1)
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z1]
        adv_tr, epsilon, mem, alpha_0, threshold, train_step, writer = args[-7:]
        z_size, x_size = z1_est.shape[1], u.shape[1]
        mem = 20
        beta = 1.0
        eps = 1e-5
        lam = 0
        alpha = 0.1
        nsteps = [0,]
        mu_est = z1_est

        def f(v, x, it):
            v = v.squeeze(-1)
            z, mu = v[:, :z_size], v[:, z_size:]
            with torch.enable_grad():
                z = Variable(z, requires_grad = True).to(x.device)
                z_out, recon_x, recon_xp, recon_fxp, diff = func(torch.cat([z, x + u], dim=1), 0, 0, only_fx_cost=True)
                mu_out = autograd.grad([z_out], [z], grad_outputs=[mu], create_graph=True)[0]
                mu_out += autograd.grad(recon_x, z)[0]
            return torch.cat([z_out, mu_out], dim=1).unsqueeze(-1)
        g = lambda v, x, it: f(v, x, it) - v

        def opt(z1, alp):
            u_adam = torch.nn.parameter.Parameter(torch.zeros_like(u))
            optimizer = torch.optim.SGD([u_adam,], lr=alp,)
            mu_est = z1*0
            nsteps[0] = 0
            for i in range(20):
                optimizer.zero_grad()
                flam = lambda z, it: f(z, u_adam, it)
                glam = lambda z, it: flam(z, it) - z
                result_info = broyden(glam, torch.cat([z1, torch.zeros_like(mu_est)], dim=1), threshold=threshold, eps=eps, ls=False, idx=True, x_size=z_size)
                z1 = result_info['result'][:, :z_size]
                mu_est = result_info['result'][:, z_size:]
                nsteps[0] += result_info['nstep']
                # compute gradients
                with torch.enable_grad():
                    z_out, recon_x, recon_xp, recon_fxp, diff = func(torch.cat([z1.squeeze(-1), u_adam + u], dim=1), 0, 0, only_fx_cost=True)  # + 0.1*(torch.mean(z**2) - 1)**2
                    u_grad = autograd.grad([x_out], [u_adam], grad_outputs=[mu_est.squeeze(-1)])[0]
                    if adv_tr:
                        u_adam.grad = u_grad.detach()/(u_grad.data.norm(dim=1, keepdim=True)+1e-8)
                    u_grad1 = u_grad.data.clone().detach()
                    
                    # optimize the latents
                    optimizer.step()
                    if adv_tr:
                        delta_norms = u_adam.data.norm(dim=1, keepdim=True)+1e-8
                        factor = epsilon / delta_norms
                        factor = torch.min(factor, torch.ones_like(delta_norms))
                        u_adam.data = u_adam.data * factor.view(-1, 1)
            return z1, u_adam, mu_est, result_info, u_grad, u_grad1


        glam = lambda z, it: func(torch.cat([z.squeeze(-1), u], dim=1), 0, 0, only_fx=True).unsqueeze(-1) - z
        if adv_tr:
            result_info = broyden(glam, z1_est, threshold=threshold, eps=eps, ls=False, idx=True, x_size=z_size)
            z1_est = result_info['result'][:, :z_size]
            u_adam = u*0
        else:
            z1_est, u_adam, mu_est, result_info, u_grad, u_grad1 = opt(z1_est, alpha_0)

        u_est = u_adam.clone().detach() + u
        mu_final = mu_est.clone().detach()
        jc_z = z1_est.clone().detach()
        jc_u = u_est.clone().detach()
        if z1_est.get_device() == 0:
            if writer is not None:
                writer.add_scalar('forward/diff', result_info['gx'][:, :z_size].norm().item(), train_step)
                writer.add_scalar('forward/lowest_step', nsteps[0], train_step)
                writer.add_scalar('forward/diff_normalized', result_info['gx'][:, :z_size].norm()/(z1_est.norm() + 1e-8), train_step)
                writer.add_scalar('forward/mu_norm', mu_est.norm().item(), train_step)
                writer.add_scalar('forward/gxmu_norm', result_info['gx'][:, -z_size:].norm().item(), train_step)
        torch.cuda.empty_cache()
        return z1_est.squeeze(-1), u_est.squeeze(-1), mu_final.squeeze(-1), jc_z, jc_u

    @staticmethod
    def forward(ctx, func, z1, u, constr=False, *args):
        nelem = sum([elem.nelement() for elem in z1])
        eps = 1e-5 * np.sqrt(nelem)
        ctx.args_len = len(args)
        ctx.constr = constr
        proj_adam = args[-8]
        with torch.no_grad():
            if constr:
                if not proj_adam:
                    z_est, u_est, mu_final, jc_z, jc_u = DEQFunc2d.anderson_gem_find_root(func, z1, u, eps, *args)
                else:
                    z_est, u_est, mu_final, jc_z, jc_u = DEQFunc2d.broyden_projected_adam(func, z1, u, eps, *args)
                return z_est, u_est, mu_final, jc_z, jc_u
            else:
                z1_est = DEQFunc2d.broyden_find_root(func, z1, u, eps, *args)
                return z1_est, z1_est*0, z1_est*0, z1_est*0, z1_est*0

    @staticmethod
    def backward(ctx, grad_z, grad_u, grad_mu, gradgx, gradugrad):
        grad_args = [None for _ in range(ctx.args_len)]
        if ctx.constr:
            grad_u = grad_u
        else:
            grad_u = None
        return (None, grad_z, grad_u, None, *grad_args)



class DEQFunc2dMeta(Function):
    """ 
    Generic DEQ-JIIO module that uses Broyden/Anderson's method to find the DEQ/JIIO/PGD 
    equilibrium state for meta learning problems
    """

    @staticmethod
    def f(func, z1, u, v, *args):
        return func(z1, u, v, *args)

    @staticmethod
    def g(func, z1, u, v, cutoffs, *args):
        z1_list = DEQFunc2dMeta.vec2list(z1, cutoffs)
        return DEQFunc2dMeta.list2vec(DEQFunc2dMeta.f(func, z1_list, u, v, *args)) - z1

    @staticmethod
    def list2vec(z1_list):
        bsz = z1_list[0].size(0)
        return torch.cat([elem.reshape(bsz, -1, 1) for elem in z1_list], dim=1)

    @staticmethod
    def vec2list(z1, cutoffs):
        bsz = z1.shape[0]
        z1_list = []
        start_idx, end_idx = 0, cutoffs[0][0] * cutoffs[0][1] * cutoffs[0][2]
        for i in range(len(cutoffs)):
            z1_list.append(z1[:, start_idx:end_idx].view(bsz, *cutoffs[i]))
            if i < len(cutoffs)-1:
                start_idx = end_idx
                end_idx += cutoffs[i + 1][0] * cutoffs[i + 1][1] * cutoffs[i + 1][2]
        return z1_list

    @staticmethod
    def broyden_find_root(func, z1, u, v, eps, *args):
        """
        Computing root for f(z1,u,v) - z1 using broyden's method 
        Note: u,v remains fixed.
        """
        bsz = z1[0].size(0)
        z1_est = DEQFunc2dMeta.list2vec(z1)
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z1]
        threshold, train_step, writer, num_ex = args[-4:]

        g = lambda x: DEQFunc2dMeta.g(func, x, u, v, cutoffs, *args)
        result_info = broyden(g, z1_est, threshold=threshold, eps=eps, name="forward")
        z1_est = result_info['result']
        nstep = result_info['nstep']
        lowest_step = result_info['lowest_step']
        diff = result_info['diff']
        r_diff = min(result_info['new_trace'][1:])
        if z1_est.get_device() == 0:
            if writer is not None:
                writer.add_scalar('forward/diff', result_info['diff'], train_step)
                writer.add_scalar('forward/nstep', result_info['nstep'], train_step)
                writer.add_scalar('forward/lowest_step', result_info['lowest_step'], train_step)
                writer.add_scalar('forward/final_trace', result_info['new_trace'][lowest_step], train_step)

        status = analyze_broyden(result_info, judge=True)
        if status:
            err = {"z1": z1}
            analyze_broyden(result_info, err=err, judge=False, name="forward", save_err=False)
        
        if threshold > 30:
            torch.cuda.empty_cache()
        return z1_est.clone().detach()

    @staticmethod
    def anderson_find_root(func, z1, u, v, eps, *args):
        """
        Computing fixed point for z1 := f(z1,u,v) using anderson acceleration. 
        Note: u,v remains fixed.
        """
        bsz = z1[0].size(0)
        z1_est = DEQFunc2dMeta.list2vec(z1)
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z1]
        threshold, train_step, writer, num_ex = args[-4:]

        f = lambda x: DEQFunc2dMeta.g(func, x, u, v, cutoffs, *args) + x
        result_info = anderson_deq(f, z1_est, threshold=threshold, eps=eps, beta=1.0, name="forward")
        z1_est = result_info['result']

        if threshold > 30:
            torch.cuda.empty_cache()
        return z1_est.clone().detach()

    @staticmethod
    def anderson_gem_find_root(func, z1, u, v, eps, *args):
        """
        Performing JIIO updates to otain fixed point updates on [z,v,mu] := F([z,v,mu],u,y) using anderson
        Note: both z1 and v are being optimized.
        """
        bsz = z1[0].size(0)
        bsz_v = v.size(0)
        z1_est = DEQFunc2dMeta.list2vec(z1).clone().detach()
        v = v.clone().detach()
        u = u.clone().detach()
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z1]
        mem, alpha_0, threshold, train_step, writer,num_ex = args[-6:]
        z_size, u_size, v_size = z1_est.shape[1], u.shape[1], v.shape[1]
        z_size_b = z_size*num_ex
        u_size_b = u_size*num_ex
        beta = 1.0
        eps = 1e-5
        lam = 1e-4
        alpha = [alpha_0[0],0.6, 0.8]
        iter_switch = max(50, int(threshold*5/8))
        interval = []
        start = time.time()
        start2 = torch.cuda.Event(enable_timing=True)
        end2 = torch.cuda.Event(enable_timing=True)
        start2.record()

        jc_v = []
        jc_z = []
        indices = np.random.randint(low=15, high=threshold, size=bsz_v)


        def f(w, it=0):
            """
            Computes updates [z,v,mu] := F([z,v,mu],y,u)
            """
            w = w.squeeze(-1)
            z, vi, mu = w[:, :z_size_b], w[:, z_size_b:z_size_b+v_size], w[:, z_size_b+v_size:]
            with torch.enable_grad():
                vi = Variable(vi, requires_grad = True).to(v.device)
                z = Variable(z, requires_grad = True).to(z.device)
                v_exp = vi.repeat_interleave(num_ex, 0)
                z_exp = z.reshape(-1, z_size)
                mu_exp = mu.reshape(-1, z_size)
                z_out, recon_x, recon_xp, recon_fxp, diff, loss = func(torch.cat([z_exp, v_exp], dim=1), u, 0, 0, only_fx_cost=True)
                mu_out, v_out = autograd.grad([z_out, recon_x], [z, vi], grad_outputs=[mu_exp, None])#, retain_graph=False, create_graph=False)
                z_out = z_out.reshape(-1, z_size_b)

                update_idx = indices==it

                if (it+1)%iter_switch==0: 
                    alpha[0] = alpha_0[1]

                mu_temp = mu + alpha[1] * (mu_out - mu)
                v_temp = vi - alpha[0] * v_out
                z_temp = z + alpha[2] * (z_out - z)

                # Store certain iterates to perform jacobian regularization
                jc_z.append(z_temp[update_idx])
                jc_v.append((v_temp)[update_idx])

                end2.record()
                torch.cuda.synchronize()
                interval.append([it, start2.elapsed_time(end2), recon_xp.mean().item(), (z_out - z).norm().item()])
            return torch.cat([z_temp, v_temp, mu_temp], dim=1).unsqueeze(-1), recon_xp.detach().clone().view(-1,num_ex).mean(dim=-1)

        g = lambda w, it: f(w, it) - w
        z1_est = z1_est.reshape(-1, z_size_b,1)
        gc.collect()
        # Perform anderson acceleration to compute fixed points on F
        result_info = anderson(f, torch.cat([z1_est, v.unsqueeze(-1), z1_est*0], dim=1), z_size, m=mem, lam=lam, threshold=threshold, eps=eps, beta=beta, acc_type='bad')
        z1_est = result_info['result'][:, :z_size_b].reshape(-1, z_size)
        v_est = result_info['result'][:, z_size_b:z_size_b + v_size]
        mu_final = result_info['result'][:, -z_size_b:].reshape(-1, z_size)
        jc_z = torch.cat(jc_z, dim=0).reshape(-1, z_size)
        jc_v = torch.cat(jc_v, dim=0)
        gc.collect()
        
        if z1_est.get_device() == 0:
            if writer is not None:
                writer.add_scalar('forward/diff', result_info['gx'][:, :z_size_b].norm().item(), train_step)
                writer.add_scalar('forward/lowest_step', result_info['nstep'].mean(), train_step)
                writer.add_scalar('forward/diff_normalized', result_info['gx'][:, :z_size_b].norm()/z1_est.norm(), train_step)
                writer.add_scalar('forward/mu_norm', result_info['result'][:, -z_size_b:].norm().item(), train_step)
                writer.add_scalar('forward/gxz_norm', result_info['gx'][:, z_size_b:z_size_b + v_size].norm().item(), train_step)
                writer.add_scalar('forward/gxz_normalized', result_info['gx'][:, z_size_b:z_size_b + v_size].norm().item()/v_est.norm().item(), train_step)
                writer.add_scalar('forward/gxmu_norm', result_info['gx'][:, -z_size_b:].norm().item(), train_step)
                writer.add_scalar('forward/gxmu_normalized', result_info['gx'][:, -z_size_b:].norm().item()/mu_final.norm().item(), train_step)
                zmean = result_info['result'][:, z_size:z_size + v_size].mean(dim=0, keepdim=True)
                zvar = ((result_info['result'][:, z_size:z_size + v_size] - zmean)**2).mean(dim=0, keepdim=True)
                writer.add_scalar('forward/zmean', zmean.norm().item(), train_step)
                writer.add_scalar('forward/zvar', zvar.norm().item(), train_step)
                
        return z1_est.squeeze(-1).clone().detach(), v_est.squeeze(-1).clone().detach(), \
        mu_final.squeeze(-1).clone().detach(), jc_z, jc_v
        
    @staticmethod
    def broyden_projected_adam(func, z1, u, v, eps, *args):
        """
        Performing Projected Adam updates on inputs v. The projection here is on the constraints z = f(z,u,v) using broyden
        """
        bsz = z1[0].size(0)
        bsz_v = v.shape[0]
        z1_est = DEQFunc2d.list2vec(z1)
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z1]
        mem, alpha_0, threshold, train_step, writer, num_ex = args[-6:]
        z_size, u_size, v_size = z1_est.shape[1], u.shape[1], v.shape[1]
        z_size_b = z_size*num_ex
        u_size_b = u_size*num_ex
        mem = 20
        beta = 1.0
        eps = 1e-5
        lam = 0
        alpha = 0.1
        nsteps = [0,]
        eps1 = 0.00001 * np.sqrt(bsz_v)

        def f(w, vi, it):
            w = w.squeeze(-1)
            z, mu = w[:, :z_size_b], w[:, -z_size_b:]
            with torch.enable_grad():
                z = Variable(z, requires_grad = True).to(z.device)
                v_exp = vi.repeat_interleave(num_ex, 0)
                z_exp = z.reshape(-1, z_size)
                mu_exp = mu.reshape(-1, z_size)
                z_out, recon_x, recon_xp, recon_fxp, diff, loss = func(torch.cat([z_exp, v_exp], dim=1), u, 0, 0, only_fx_cost=True)
                mu_out = autograd.grad([z_out, recon_x], [z], grad_outputs=[mu_exp, None])[0]#, create_graph=True)[0]
                z_out = z_out.reshape(-1, z_size_b)
            return torch.cat([z_out, mu_out], dim=1).unsqueeze(-1)
        g = lambda w, vi, it: f(w, vi, it) - w

        v_adam = torch.nn.parameter.Parameter(v.detach())
        optimizer = torch.optim.Adam([v_adam,], lr=alpha,)
        z1_est = z1_est.reshape(-1, z_size_b,1)
        mu_est = z1_est*0
        z_list = []
        v_list = []
        indices = np.random.randint(low=0, high=20, size=z1_est.shape[0])

        for i in range(20):
            # compute objective
            optimizer.zero_grad()
            # g = lambda x: func(torch.cat([x, u_adam], dim=1), 0, 0, only_fx=True)[1]        
            flam = lambda x, it: f(x, v_adam, it)
            glam = lambda x, it: flam(x, it) - x
            # result_info = anderson(flam, torch.cat([z1_est, mu_est], dim=1), x_size, m=mem, lam=lam, threshold=threshold, eps=eps, beta=beta)
            result_info = broyden(glam, torch.cat([z1_est, mu_est], dim=1), threshold=threshold, eps=eps, ls=False, idx=True, x_size=z_size)
            z1_est = result_info['result'][:, :z_size_b]
            mu_est = result_info['result'][:, -z_size_b:]
            nsteps[0] += result_info['nstep']
            z_exp = z1_est.reshape(-1, z_size)
            mu_exp = mu_est.reshape(-1, z_size)
            update_idx = indices==i
            z_list.append(z1_est[update_idx].clone().detach())
            v_list.append(v_adam[update_idx].data.clone().detach())

            # compute gradients
            with torch.enable_grad():
                v_exp = v_adam.repeat_interleave(num_ex, 0)
                z_out, recon_x, recon_xp, recon_fxp, diff, loss = func(torch.cat([z_exp, v_exp], dim=1), u, 0, 0, only_fx_cost=True)  # + 0.1*(torch.mean(z**2) - 1)**2

                # compute updates
                v_grad = autograd.grad([z_out, recon_x], [v_adam], grad_outputs=[mu_exp, None])[0]
                v_adam.grad = v_grad.detach()
                v_grad1 = v_grad.data.clone().detach()

                # optimize the inputs
                optimizer.step()
        update_idx = indices>i
        z_list.append(z1_est[update_idx].clone().detach())
        v_list.append(v_adam[update_idx].data.clone().detach())
        v_est = v_adam.clone().detach()
        mu_final = mu_exp.clone().detach()#result_info['result'][:, -x_size:]
        z1_est = z_exp.clone().detach()

        z_list = torch.cat(z_list, dim=0).squeeze()
        v_list = torch.cat(v_list, dim=0).squeeze()
        z_list = z_list.reshape(-1, z_size)
        if z1_est.get_device() == 0:
            if writer is not None:
                # print("gx_final", result_info['gx'][:, :z_size].norm().item(), v_grad1.norm().item())
                writer.add_scalar('forward/diff', result_info['gx'][:, :z_size].norm().item(), train_step)
                writer.add_scalar('forward/lowest_step', nsteps[0], train_step)
                writer.add_scalar('forward/diff_normalized', result_info['gx'][:, :z_size].norm()/(z1_est.norm() + 1e-8), train_step)
                writer.add_scalar('forward/mu_norm', mu_est.norm().item(), train_step)
                writer.add_scalar('forward/gxz_norm', v_grad1.norm().item(), train_step)
                writer.add_scalar('forward/gxmu_norm', result_info['gx'][:, -z_size:].norm().item(), train_step)
        torch.cuda.empty_cache()
        return z1_est.squeeze(-1), v_est.squeeze(-1), mu_final.squeeze(-1), z_list, v_list

    @staticmethod
    def forward(ctx, func, z1, u, v, constr=False, *args):
        nelem = sum([elem.nelement() for elem in z1])
        eps = 1e-5 * np.sqrt(nelem)
        ctx.args_len = len(args)
        ctx.constr = constr
        proj_adam = args[-7]
        with torch.no_grad():
            if constr:
                if not proj_adam:
                    z_est, v_est, mu_final, jc_z, jc_v = DEQFunc2dMeta.anderson_gem_find_root(func, z1, u, v, eps, *args)
                else:
                    z_est, v_est, mu_final, jc_z, jc_v= DEQFunc2dMeta.broyden_projected_adam(func, z1, u, v, eps, *args)
                ctx.id = 1
                return z_est, v_est, mu_final, jc_z, jc_v
            else:
                z1_est = DEQFunc2dMeta.broyden_find_root(func, z1, u, v, eps, *args)
                return z1_est, v.clone().detach(), z1_est*0, z1_est*0, z1_est*0

    @staticmethod
    def backward(ctx, grad_z, grad_v, grad_mu, gradgx, gradugrad):
        grad_args = [None for _ in range(ctx.args_len)]
        if ctx.constr:
            grad_v = grad_v
        else:
            grad_u = None
        return (None, grad_z, None, grad_v, None, *grad_args)


    
class DEQModule2d(nn.Module):
    def __init__(self, func, func_copy):
        super(DEQModule2d, self).__init__()
        self.func = func
        self.func_copy = func_copy

    def forward(self, z1s, us, z0, **kwargs):
        raise NotImplemented

    class Backward(Function):
        
        @staticmethod
        def forward(ctx, func_copy, z1, u, mu_final, *args):
            ctx.save_for_backward(z1)
            ctx.u = u
            ctx.mu_final = mu_final
            ctx.func = func_copy
            ctx.args = args
            return z1.clone()

        @staticmethod
        def backward(ctx, grad):
            
            bsz, d_model, seq_len = grad.size()
            grad = grad.clone()
            z1, = ctx.saved_tensors
            u = ctx.u
            factor = sum(ue.nelement() for ue in u) // z1.nelement()
            cutoffs = [(elem.size(1) // factor, elem.size(2), elem.size(3)) for elem in u]
            args = ctx.args
            threshold, train_step, writer = args[-3:]
            size_avg = args[-4]
            task = args[0]
            if task == 'meta': 
                v = args[1]
                v_temp = v.clone()

            func = ctx.func
            mu_final = ctx.mu_final.clone().detach()/(size_avg)
            z1_temp = z1.clone().detach().requires_grad_()
            u_temp = [elem.clone().detach() for elem in u]
            args_temp = args[:-1]
            
            with torch.enable_grad():
                if task == 'meta':
                    y = DEQFunc2dMeta.g(func, z1_temp, u_temp, v_temp, cutoffs, *args_temp)
                else:
                    y = DEQFunc2d.g(func, z1_temp, u_temp, cutoffs, *args_temp)

            def g(x):
                y.backward(x, retain_graph=True)  # Retain for future calls to g
                res = z1_temp.grad + grad
                z1_temp.grad.zero_()
                return res

            eps = 2e-10 * np.sqrt(bsz * seq_len * d_model)
            
            # We don't want to reuse mu from the forward pass for meta-learning experiments
            if not task == 'meta':
                dl_df_est = mu_final.unsqueeze(-1).detach().clone()
            else:
                dl_df_est = torch.zeros_like(grad)
            result_info = broyden(g, dl_df_est, threshold=threshold, eps=eps, name="backward")
            dl_df_est = result_info['result']
            nstep = result_info['nstep']
            lowest_step = result_info['lowest_step']
            
            if dl_df_est.get_device() == 0:
                if writer is not None:
                    writer.add_scalar('backward/diff', result_info['diff'], train_step)
                    writer.add_scalar('backward/nstep', result_info['nstep'], train_step)
                    writer.add_scalar('backward/lowest_step', result_info['lowest_step'], train_step)
                    writer.add_scalar('backward/final_trace', result_info['new_trace'][lowest_step], train_step)

            status = analyze_broyden(result_info, judge=True)
            if status:
                err = {"z1": z1}
                analyze_broyden(result_info, err=err, judge=False, name="backward", save_err=False)

            if threshold > 30:
                torch.cuda.empty_cache()

            y.backward(torch.zeros_like(dl_df_est), retain_graph=False)

            grad_args = [None for _ in range(len(args))]
            return (None, dl_df_est, None, None, *grad_args)


