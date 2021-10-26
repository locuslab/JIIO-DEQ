# Modified based on the DEQ repo.

import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import numpy as np 
import pickle
import sys
import os
from scipy.optimize import root
import time
from termcolor import colored
from scipy.optimize import SR1 as scipy_sr1


def _safe_norm(v):
    if not torch.isfinite(v).all():
        return np.inf
    return torch.norm(v)


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    ite = 0
    phi_a0 = phi(alpha0)    # First do an update with step size 1
    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0, ite

    # Otherwise, compute the minimizer of a quadratic interpolant
    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)
        ite += 1

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2, ite

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1, ite


def line_search(update, x0, g0, g, nstep=0, on=True):
    """
    `update` is the propsoed direction of update.

    Code adapted from scipy.
    """
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0)**2]
    s_norm = torch.norm(x0) / torch.norm(update)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]    # If the step size is so small... just return something
        
        x_est = x0 + s * update
        g0_new = g(x_est)
        phi_new = _safe_norm(g0_new)**2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new
    
    if on:
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-12)
    if (not on) or s is None:
        s = 1.0
        ite = 0
    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)
    return x_est, g0_new, x_est - x0, g0_new - g0, ite

def init_fn(x, z_size):
    # mu, z, x = x[:, :z_size], x[:, z_size:2*z_size], x[:, 2*z_size:]
    mu, z, x = x[:, :z_size], x[:, z_size:-z_size], x[:, -z_size:]
    return torch.cat([-mu, - z,  -x], dim=1), torch.cat([mu, z, x], dim=1)

def rmatvec(part_Us, part_VTs, x, init=1):
    # Compute x^T(-I + UV^T)
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if init >1:
        init_x, x = init_fn(x, init)
    else:
        init_x = -init*x
    if part_Us.nelement() == 0:
        return init_x
    xTU = torch.einsum('bij, bijd -> bd', x, part_Us)   # (N, threshold)
    return init_x + torch.einsum('bd, bdij -> bij', xTU, part_VTs)    # (N, 2d, L'), but should really be (N, 1, (2d*L'))


def matvec(part_Us, part_VTs, x, init=1):
    # Compute (-I + UV^T)x
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if init>1:
        init_x, x = init_fn(x, init)
    else:
        init_x = -init*x
    if part_Us.nelement() == 0:
        return init_x
    VTx = torch.einsum('bdij, bij -> bd', part_VTs, x)  # (N, threshold)
    return init_x + torch.einsum('bijd, bd -> bij', part_Us, VTx)     # (N, 2d, L'), but should really be (N, (2d*L'), 1)


def broyden(g, x0, threshold, eps, init=1, ls=False, name="unknown", idx=False, x_size=None, printi=True):
    bsz, total_hsize, n_elem = x0.size()
    dev = x0.device
    
    x_est = x0           # (bsz, 2d, L')
    if idx:
        gx = g(x_est, 0)        # (bsz, 2d, L')
    else:
        gx = g(x_est)
    nstep = 0
    tnstep = 0
    LBFGS_thres = min(threshold, 20)
    
    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(bsz, total_hsize, n_elem, LBFGS_thres).to(dev)
    VTs = torch.zeros(bsz, LBFGS_thres, total_hsize, n_elem).to(dev)
    update = -matvec(Us[:,:,:,:nstep], VTs[:,:nstep], gx, init)# -gx
    new_objective = init_objective = torch.norm(gx).item()
    prot_break = False
    trace = [init_objective]
    new_trace = [-1]
    
    # To be used in protective breaks
    protect_thres = 1e6 * n_elem
    lowest = new_objective
    lowest_xest, lowest_gx, lowest_step = x_est, gx, nstep
    
    while new_objective >= eps and nstep < threshold:
        if idx:
            g1 = lambda x: g(x, nstep)
        else:
            g1 = g
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g1, nstep=nstep, on=ls)
        nstep += 1
        tnstep += (ite+1)
        new_objective = torch.norm(gx).item()
        trace.append(new_objective)
        try:
            new2_objective = torch.norm(delta_x).item() / (torch.norm(x_est - delta_x).item())   # Relative residual
        except:
            new2_objective = torch.norm(delta_x).item() / (torch.norm(x_est - delta_x).item() + 1e-9)
        new_trace.append(new2_objective)
        if new_objective < lowest:
            lowest_xest, lowest_gx = x_est.clone().detach(), gx.clone().detach()
            lowest = new_objective
            lowest_step = nstep
        if new_objective < eps:
            break
        if new_objective < 3*eps and nstep > 30 and np.max(trace[-30:]) / np.min(trace[-30:]) < 1.3:
            # if there's hardly been any progress in the last 30 steps
            break
        if new_objective > init_objective * protect_thres:
            prot_break = True
            break

        part_Us, part_VTs = Us[:,:,:,:(nstep-1)], VTs[:,:(nstep-1)]
        # uncomment depending on good broyden vs bad broyden : both usually work.
        vT = delta_gx                                     # good broyden
        # vT = rmatvec(part_Us, part_VTs, delta_x, init)  # bad broyden 
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx, init)) / torch.einsum('bij, bij -> b', vT, delta_gx)[:,None,None]
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:,(nstep-1) % LBFGS_thres] = vT
        Us[:,:,:,(nstep-1) % LBFGS_thres] = u
        update = -matvec(Us[:,:,:,:nstep], VTs[:,:nstep], gx, init)

    Us, VTs = None, None
    return {"result": lowest_xest,
            "nstep": nstep,
            "tnstep": tnstep,
            "lowest_step": lowest_step,
            "diff": torch.norm(lowest_gx).item(),
            "diff_detail": torch.norm(lowest_gx, dim=1),
            "prot_break": prot_break,
            "trace": trace,
            "new_trace": new_trace,
            "eps": eps,
            "threshold": threshold,
            "gx": lowest_gx}

def anderson(f, x0, x_size, m=5, lam=1e-4, threshold=50, eps=1e-5, stop_mode='rel', beta=0.8, acc_type='good', **kwargs):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, L = x0.shape
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    X = torch.zeros(bsz, m, d*L, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*L, dtype=x0.dtype, device=x0.device)
    fi, cost = f(x0, 0)
    fi = fi.reshape(bsz, -1)
    X[:,0], F[:,0] = x0.reshape(bsz, -1), fi
    fi, cost = f(F[:,0].reshape_as(x0), 1)
    fi = fi.reshape(bsz, -1)
    X[:,1], F[:,1] = F[:,0], fi
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1

    trace_dict = {'abs': [],
                  'rel': []}

    lowest_dict = {'abs': 1e12*torch.ones_like(F[:,0,0]),
                   'rel': 1e12*torch.ones_like(F[:,0,0])}
    lowest_step_dict = {'abs': np.ones(bsz),
                        'rel': np.ones(bsz)}

    lowest_xest, lowest_gx =  X[:,1].view_as(x0).clone().detach(), X[:,1].view_as(x0).clone().detach()*0

    lowest_cost = cost
    time1_ = []
    time2_ = []
    time3_ = []

    for k in range(2, threshold):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        if acc_type == 'good':
            if k>2:
                H[:,(k-1)%m+1,1:n+1] = torch.bmm(X[:,(k-1)%m].unsqueeze(1),G.transpose(1,2)).squeeze(1)
                H[:,1:n+1,(k-1)%m+1] = torch.bmm(X[:,:n],G[:,(k-1)%m].unsqueeze(1).transpose(1,2)).squeeze(-1)
            else:
                H[:,1:n+1,1:n+1] = torch.bmm(X[:,:n],G.transpose(1,2))
        else:
            if k>2:
                H[:,(k-1)%m+1,1:n+1] = torch.bmm(G[:,(k-1)%m].unsqueeze(1),G.transpose(1,2)).squeeze(1)
                H[:,1:n+1,(k-1)%m+1] = torch.bmm(G[:,:n],G[:,(k-1)%m].unsqueeze(1).transpose(1,2)).squeeze(-1)
            else:
                H[:,1:n+1,1:n+1] = torch.bmm(G[:,:n],G.transpose(1,2))

        # Could just do alpha = ...
        # But useful when working with some weird scenarios. Helps with ill-conditioned H
        while True:
            try:
                alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])   # (bsz x n)
                break
            except:
                lam = lam*10
                H[:,1:n+1,1:n+1] += lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]

        alpha = alpha[0][:, 1:n+1, 0]
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]

        fi, cost = f(X[:,k%m].reshape_as(x0), k)
        F[:,k%m] = fi.reshape(bsz, -1)
        gx = (F[:,k%m] - X[:,k%m]).view_as(x0)
        diff_x = gx[:, :x_size].norm().item()
        abs_diff = gx.norm(dim=1).squeeze(dim=-1)
        rel_diff = abs_diff / (1e-5 + F[:,k%m].norm(dim=1))
        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}

        # forward pass criterion : always accept updates for the first 10 steps
        #                        : then make a tradeoff between lower cost and lower kkt residual
        dict_mask = torch.logical_or(diff_dict[stop_mode] < lowest_dict[stop_mode], torch.tensor(k<10))
        mask = torch.logical_or(torch.logical_or(diff_dict[stop_mode] < lowest_dict[stop_mode], torch.tensor(k<10)),torch.logical_and(cost < lowest_cost, diff_dict[stop_mode] < 1.3*lowest_dict[stop_mode]))
        lowest_xest[mask] = X[mask,k%m].clone().detach().unsqueeze(-1)
        lowest_gx[mask] = gx[mask].clone().detach()
        lowest_dict[stop_mode][dict_mask] = diff_dict[stop_mode][dict_mask]
        lowest_step_dict[stop_mode][mask.cpu().numpy()] = k
        lowest_cost[mask] = cost[mask].clone().detach()

    out = {"result": lowest_xest,
           "gx" : lowest_gx,
           "lowest": lowest_dict[stop_mode],
           "nstep": lowest_step_dict[stop_mode],
           "prot_break": False,
           "abs_trace": trace_dict['abs'],
           "rel_trace": trace_dict['rel'],
           "eps": eps,
           "threshold": threshold}
    X = F = None
    return out

def anderson_deq(f, x0, m=5, lam=1e-4, threshold=50, eps=1e-5, stop_mode='rel', beta=0.8, **kwargs):
    """ 
    Anderson acceleration for fixed point iteration.
    Experimenting with other stopping criterion 
    """
    bsz, d, L = x0.shape
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    X = torch.zeros(bsz, m, d*L, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*L, dtype=x0.dtype, device=x0.device)
    fi = f(x0)
    fi = fi.reshape(bsz, -1)
    X[:,0], F[:,0] = x0.reshape(bsz, -1), fi
    fi = f(F[:,0].reshape_as(x0))
    fi = fi.reshape(bsz, -1)
    X[:,1], F[:,1] = F[:,0], fi
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1

    trace_dict = {'abs': [],
                  'rel': []}

    lowest_dict = {'abs': 1e12*torch.ones_like(F[:,0,0]),
                   'rel': 1e12*torch.ones_like(F[:,0,0])}
    lowest_step_dict = {'abs': np.ones(bsz),
                        'rel': np.ones(bsz)}

    lowest_xest, lowest_gx =  X[:,1].view_as(x0).clone().detach(), X[:,1].view_as(x0).clone().detach()*0

    # lowest_cost = cost
    time1_ = []
    time2_ = []
    time3_ = []

    for k in range(2, threshold):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        try:
            if k>2:
                H[:,(k-1)%m+1,1:n+1] = torch.bmm(X[:,(k-1)%m].unsqueeze(1),G.transpose(1,2)).squeeze(1)
                H[:,1:n+1,(k-1)%m+1] = torch.bmm(X[:,:n],G[:,(k-1)%m].unsqueeze(1).transpose(1,2)).squeeze(-1)
            else:
                H[:,1:n+1,1:n+1] = torch.bmm(X[:,:n],G.transpose(1,2))
        except:
            ipdb.set_trace()
        while True:
            try:
                alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])   # (bsz x n)
                break
            except:
                lam = lam*10
                H[:,1:n+1,1:n+1] += lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]

        alpha = alpha[0][:, 1:n+1, 0]
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]

        fi = f(X[:,k%m].reshape_as(x0))
        F[:,k%m] = fi.reshape(bsz, -1)
        gx = (F[:,k%m] - X[:,k%m]).view_as(x0)
        abs_diff = gx.norm(dim=1).squeeze(dim=-1)
        rel_diff = abs_diff / (1e-5 + F[:,k%m].norm(dim=1))
        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}
        mask = torch.logical_or(diff_dict[stop_mode] < lowest_dict[stop_mode], torch.tensor(k<8))
        lowest_xest[mask] = X[mask,k%m].clone().detach().unsqueeze(-1)
        lowest_gx[mask] = gx[mask].clone().detach()
        lowest_dict[stop_mode][mask] = diff_dict[stop_mode][mask]
        lowest_step_dict[stop_mode][mask.cpu().numpy()] = k
        # lowest_cost[mask] = cost[mask].clone().detach()

    out = {"result": lowest_xest,
           "gx" : lowest_gx,
           "lowest": lowest_dict[stop_mode],
           "nstep": lowest_step_dict[stop_mode],
           "prot_break": False,
           "abs_trace": trace_dict['abs'],
           "rel_trace": trace_dict['rel'],
           "eps": eps,
           "threshold": threshold}
    X = F = None
    return out

def analyze_broyden(res_info, err=None, judge=True, name='forward', training=True, save_err=True):
    """
    For debugging use only :-)
    """
    res_est = res_info['result']
    nstep = res_info['nstep']
    diff = res_info['diff']
    diff_detail = res_info['diff_detail']
    prot_break = res_info['prot_break']
    trace = res_info['trace']
    eps = res_info['eps']
    threshold = res_info['threshold']
    if judge:
        return nstep >= threshold or (nstep == 0 and (diff != diff or diff > eps)) or prot_break or torch.isnan(res_est).any()
    
    assert (err is not None), "Must provide err information when not in judgment mode"
    prefix, color = ('', 'red') if name == 'forward' else ('back_', 'blue')
    eval_prefix = '' if training else 'eval_'
    
    # Case 1: A nan entry is produced in Broyden
    if torch.isnan(res_est).any():
        msg = colored(f"WARNING: nan found in Broyden's {name} result. Diff: {diff}", color)
        print(msg)
        if save_err: pickle.dump(err, open(f'{prefix}{eval_prefix}nan.pkl', 'wb'))
        return (1, msg, res_info)
        
    # Case 2: Unknown problem with Broyden's method (probably due to nan update(s) to the weights)
    if nstep == 0 and (diff != diff or diff > eps):
        msg = colored(f"WARNING: Bad Broyden's method {name}. Why?? Diff: {diff}. STOP.", color)
        print(msg)
        if save_err: pickle.dump(err, open(f'{prefix}{eval_prefix}badbroyden.pkl', 'wb'))
        return (2, msg, res_info)
        
    # Case 3: Protective break during Broyden (so that it does not diverge to infinity)
    if prot_break and np.random.uniform(0,1) < 0.05:
        msg = colored(f"WARNING: Hit Protective Break in {name}. Diff: {diff}. Total Iter: {len(trace)}", color)
        print(msg)
        if save_err: pickle.dump(err, open(f'{prefix}{eval_prefix}prot_break.pkl', 'wb'))
        return (3, msg, res_info)
        
    return (-1, '', res_info)

