# Modified based on the DEQ repo.
import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import numpy as np
# import ipdb

import sys
sys.path.append("../../")
from modules.deq2d import *

class MDEQWrapper(DEQModule2d):
    def __init__(self, func, func_copy):
        """
        MDEQ wrapper : Calls functions to compute the forward and backward fixed points for a MDEQ
        """
        super(MDEQWrapper, self).__init__(func, func_copy)
    
    def forward(self, z1, u, **kwargs):
        train_step = kwargs.get('train_step', -1)
        threshold = kwargs.get('threshold', 30)
        writer = kwargs.get('writer', None)

        if u is None:
            raise ValueError("Input injection is required.")
        cutout = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z1]
        new_z1, _, _, _, _ = DEQFunc2d.apply(self.func, z1, u, False, False, 0., 0., 0., 0., threshold, train_step, writer)
        new_z1 = Variable(new_z1.squeeze(-1), requires_grad=True)
        new_z1_copy = new_z1
        new_z1 = DEQFunc2d.vec2list(new_z1, cutout)
        jac_loss = torch.tensor([0.]).cuda()
        if self.training:
            new_z1 = DEQFunc2d.list2vec(DEQFunc2d.f(self.func, new_z1, u, threshold, train_step))
            jac_loss = self.jac_loss_estimate(new_z1.squeeze(-1), new_z1_copy)
            new_z1 = self.Backward.apply(self.func_copy, new_z1, u, new_z1.squeeze(-1)*0, 'pgd', 10, threshold, train_step, writer)
            new_z1 = DEQFunc2d.vec2list(new_z1, cutout)
        self.jac_loss = jac_loss
        return new_z1

    def jac_loss_estimate(self, f0, z0, create_graph=True):
        """ Computes the jacobian regularization using the Hutchinson estimator"""
        bsz = z0.shape[0]
        vecs = 2    # z0 already has a batch dimension, which is stochastic!
        result = 0
        for i in range(vecs):
            v = torch.randn(*z0.shape).to(z0)
            vJ = torch.autograd.grad(f0, z0, v, retain_graph=True, create_graph=create_graph)[0]
            result += vJ.norm()**2
        return result / vecs / np.prod(z0.shape)

class MDEQOptWrapper(DEQModule2d):
    def __init__(self, enc, enc_copy, dec, dec_copy):
        """
        MDEQ JIIO wrapper for Decoders: Calls functions to perform input/latent optimization
        and backward fixed points for a MDEQ Decoder. Also defines helper functions for the 
        optimization (e.g defining objectives, computing forward pass through MDEQ layer etc.)
        """
        super(MDEQOptWrapper, self).__init__(dec, dec_copy)
        self.enc = enc
        self.dec = dec
        self.enc_copy = enc_copy
        self.dec_copy = dec_copy

    def forward(self, u, z1_list, targ, output_layer, get_uprimelist, **kwargs):
        """Call functions to perform input/latent optimization and backward fixed points"""

        train_step = kwargs.get('train_step', -1)
        threshold = kwargs.get('threshold', 30)
        writer = kwargs.get('writer', None)
        alpha_0 = kwargs.get('alpha_0', 0.01)
        mem = kwargs.get('mem', 40)
        epsilon = kwargs.get('epsilon', 1.)
        traj_reg = kwargs.get('traj_reg', False)
        jiio_thres = kwargs.get('jiio_thres', 100)
        inv_prob = kwargs.get('inv_prob', 'reconstruction')
        adv_tr = False
        self.get_uprimelist = get_uprimelist
        self.output_layer = output_layer
        self.ce_cost = False
        self.bwd_pass = False
        self.criterion = lambda output, target, dim: ((output - target)**2).mean(dim = dim)

        if isinstance(u, list):
            self.only_dec = False
            self.u_cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in u]
            u = DEQFunc2d.list2vec(u).squeeze(-1)
        else:
            self.only_dec = True
            self.u_cutoffs = []
        
        
        if z1_list is None:
            raise ValueError("Input injection is required.")

        self.z_cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z1_list]
        
        self.u = u
        self.bsz, self.u_size = u.size()
        self.imgsize = targ.shape
        self.mask = torch.ones(self.imgsize).to(u.device).to(u.dtype)

        # targets defined based on the specific reconstruction task.
        if inv_prob=='inpainting':
            center = np.random.randint(self.imgsize[-1]-20, size=2*self.bsz)+10
            for i in range(self.bsz):
                self.mask[i, :, center[2*i]-10:center[2*i]+10, center[2*i+1]-10:center[2*i+1]+10]*=0
        elif inv_prob=='denoising02':
            self.targ = targ.view(self.bsz, -1) 
            self.targ_orig = targ.view(self.bsz, -1).clone()
            self.targ += torch.normal(mean=torch.zeros_like(self.targ), std=torch.zeros_like(self.targ) + 0.2)
        elif inv_prob=='denoising04':
            self.targ = targ.view(self.bsz, -1) 
            self.targ_orig = targ.view(self.bsz, -1).clone()
            self.targ += torch.normal(mean=torch.zeros_like(self.targ), std=torch.zeros_like(self.targ) + 0.4)
        else:
            self.targ = targ.view(self.bsz, -1) 
            self.targ_orig = targ.view(self.bsz, -1).clone()
        self.mask = self.mask.view(self.bsz, -1)

        # Call function to perform input optimization
        new_z1, new_u, mu_final, z_traj, u_traj = DEQFunc2d.apply(self.diffunc, z1_list, u, True, False, adv_tr, epsilon, mem, alpha_0, jiio_thres, train_step, writer)

        new_z1 = Variable(new_z1, requires_grad=True)
        new_z1_copy = new_z1
        new_z1_list = DEQFunc2d.vec2list(new_z1_copy, self.z_cutoffs)
        jac_loss = torch.tensor([0]).to(new_z1)
        new_u_jc = u_traj.clone().detach()
        new_z1_jc = Variable(z_traj, requires_grad=True)
        new_z1_jc_list = DEQFunc2d.vec2list(new_z1_jc, self.z_cutoffs)

        if self.training:
            u_list = self.get_uprimelist(new_u, self.u_cutoffs, self.bwd_pass)
            new_z1 = DEQFunc2d.list2vec(DEQFunc2d.f(self.func, new_z1_list, u_list, threshold, train_step))

            # Compute Jacobian regularization
            if traj_reg:
                ujc_list = self.get_uprimelist(new_u_jc, self.u_cutoffs, self.bwd_pass)
                new_z1_jc1 = DEQFunc2d.list2vec(DEQFunc2d.f(self.func, new_z1_jc_list, ujc_list, threshold, train_step))
                
                jac_loss = self.jac_loss_estimate(new_z1_jc1.squeeze(-1), new_z1_jc)
            else:
                jac_loss = self.jac_loss_estimate(new_z1.squeeze(-1), new_z1_copy)

            self.bwd_pass = True

            # Save variables for Backward
            new_z1 = self.Backward.apply(self.func_copy, new_z1, u_list, mu_final,'gen', np.prod(targ.shape), threshold, train_step, writer)
            new_z1_list = DEQFunc2d.vec2list(new_z1, self.z_cutoffs)
        self.new_u = new_u
        self.mask_out = self.mask.view(self.imgsize)
        self.z_traj = z_traj
        self.u_traj = u_traj
        self.jac_loss = jac_loss
        self.interval = u_traj.cpu().numpy()
        return new_z1_list

    def jac_loss_estimate(self, f0, z0, create_graph=True):
        """ Computes the jacobian regularization using the Hutchinson estimator"""
        bsz = z0.shape[0]
        vecs = 2    # z0 already has a batch dimension, which is stochastic!
        result = 0
        for i in range(vecs):
            v = torch.randn(*z0.shape).to(z0)
            vJ = torch.autograd.grad(f0, z0, v, retain_graph=True, create_graph=create_graph)[0]
            result += vJ.norm()**2
        return result / vecs / np.prod(z0.shape)

    def g(self, z, u, targ, fz, dim=(0,1)):
        """
        Computes the Cost || y - y_targ ||^2
        and also computes || f(x, z) - y_targ ||^2 
        """
        y = self.output_layer(z, self.z_cutoffs, self.bwd_pass).view(self.bsz, -1)
        fx = self.output_layer(fz, self.z_cutoffs, self.bwd_pass).view(self.bsz, -1)
        
        recon_x_sq = torch.mean((y - targ)**2, dim=dim)
        recon_fx_sq = torch.mean((y - self.targ_orig)**2, dim=dim)
        recon_x_vec = self.criterion(y*self.mask, targ*self.mask, dim=1)
        recon_x = recon_x_vec.mean()
        recon_fx = self.criterion(fx*self.mask, targ*self.mask, dim)
        if torch.isnan(recon_x.norm()):
            ipdb.set_trace()
        if len(dim)==1 and self.ce_cost:
            recon_x*=self.bsz
            recon_fx*=self.bsz
        return recon_x, recon_fx, recon_x_sq, recon_fx_sq, (fx*self.mask - targ*self.mask), recon_x_vec

    def diffunc(self, uz, mu, rho, return_all=False, vec=False, only_cost=False, only_fx=False, only_fx_cost=False):
        """
        Computes the constraint and the cost
        constraint = diff_dec = f(z, u) - z
        cost = || y - y_targ ||^2
        """
        z, u = uz[:, :-self.u_size], uz[:, -self.u_size:]
        kl_loss = 0
        if only_cost:
            y = self.output_layer(z, self.z_cutoffs, self.bwd_pass).view(self.bsz, -1)
            recon_x = self.criterion(y*self.mask, self.targ*self.mask, (0,1))
            return recon_x
        start = time.time()
        diff_dec, fz = self.diff_dec(u, z)
        if only_fx_cost:
            recon_x, recon_fx, recon_x_sq, recon_fx_sq, difffx, diffx = self.g(z, u, self.targ, fz)
            return fz, recon_x*np.prod(difffx.shape), recon_x, recon_fx_sq, diffx
        if only_fx:
            return fz, diff_dec

    def diff_dec(self, u, z):
        """
        Compute the constraints = f(z, u) - z for the decoder
        """

        z_list_prime = DEQFunc2d.vec2list(z, self.z_cutoffs)
        u_list_prime = self.get_uprimelist(u, self.u_cutoffs, self.bwd_pass)
        if self.bwd_pass:
            fz_list = self.dec_copy(z_list_prime, u_list_prime)
        else:
            fz_list = self.dec(z_list_prime, u_list_prime)
        fz = DEQFunc2d.list2vec(fz_list).squeeze(-1)

        return (fz - z).reshape(self.bsz, -1), fz

class MDEQClsWrapper(DEQModule2d):
    def __init__(self, func, func_copy):
        """
        MDEQ JIIO wrapper for input optimization in MDEQ classifiers: Calls functions to perform 
        input optimization and backward fixed points for a MDEQ classifier. Also defines helper
        functions for the optimization (e.g defining objectives, computing forward pass 
        through MDEQ layer etc.)
        """        
        super(MDEQClsWrapper, self).__init__(func, func_copy)
        self.jac_loss = torch.tensor([0.]).cuda()

    def forward(self, u, z1_list, targ, output_layer, get_ulist, **kwargs):
        """Call functions to perform input optimization and backward fixed points"""

        train_step = kwargs.get('train_step', -1)
        threshold = kwargs.get('threshold', 30)
        writer = kwargs.get('writer', None)
        proj_adam = kwargs.get('proj_adam',False)
        alpha_0 = kwargs.get('alpha_0', 0.1)
        mem = kwargs.get('mem', 20)
        epsilon = kwargs.get('epsilon', 1.)
        traj_reg = kwargs.get('traj_reg', False)
        jiio_thres = kwargs.get('jiio_thres', 80)
        adv_tr = True

        self.get_ulist = get_ulist
        self.output_layer = output_layer
        self.bwd_pass = False
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        if isinstance(u, list):
            self.only_dec = False
            self.u_cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in u]
            u = DEQFunc2d.list2vec(u).squeeze(-1)
        else:
            self.only_dec = True
            self.u_cutoffs = []
                
        if u is None:
            raise ValueError("Input injection is required.")

        self.u_cutoffs = self.z_cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z1_list]
        
        self.u = u
        self.bsz, self.u_size = u.size()
        self.targ = targ.view(-1)

        # Call function to perform input optimization
        new_z1, new_u, mu_final, z_traj, u_traj = DEQFunc2d.apply(self.diffunc, z1_list, u, True, proj_adam, adv_tr, epsilon, mem, alpha_0, jiio_thres, train_step, writer)
        new_z1 = Variable(new_z1, requires_grad=True)
        new_z1_copy = new_z1

        # At test time project onto the constraints after performing JIIO just to be sure.
        if not self.training and not proj_adam:
            new_z1_copy, _, _, _, _ = DEQFunc2d.apply(self.func, DEQFunc2d.vec2list(new_z1*0, self.z_cutoffs), self.get_ulist(new_u, self.u_cutoffs), False, False, adv_tr, 0., mem, alpha_0, threshold, train_step, writer)
        new_z1_list = DEQFunc2d.vec2list(new_z1_copy, self.z_cutoffs)
        jac_loss = torch.tensor([0.]).to(new_z1)
        new_u_jc = u_traj.clone().detach()
        new_z1_jc = Variable(z_traj, requires_grad=True)
        new_z1_jc_list = DEQFunc2d.vec2list(new_z1_jc, self.z_cutoffs)
        

        if self.training:
            u_list = self.get_ulist(new_u, self.u_cutoffs, self.bwd_pass)
            new_z1 = DEQFunc2d.list2vec(DEQFunc2d.f(self.func, new_z1_list, u_list, threshold, train_step))

            # Compute Jacobian regularization
            if traj_reg and not proj_adam:
                ujc_list = self.get_ulist(new_u_jc, self.u_cutoffs, self.bwd_pass)
                new_z1_jc1 = DEQFunc2d.list2vec(DEQFunc2d.f(self.func, new_z1_jc_list, ujc_list, threshold, train_step))
            if not traj_reg or proj_adam:
                jac_loss = self.jac_loss_estimate(new_z1.squeeze(-1), new_z1_copy)
            else:
                jac_loss = self.jac_loss_estimate(new_z1_jc1.squeeze(-1), new_z1_jc)

            self.bwd_pass = True
            new_z1 = self.Backward.apply(self.func_copy, new_z1, u_list, -mu_final, 'adv', self.bsz*10, threshold, train_step, writer)
            new_z1_list = DEQFunc2d.vec2list(new_z1, self.z_cutoffs)
        self.new_z1 = new_z1
        self.z_traj = z_traj
        self.u_traj = u_traj
        self.jac_loss = jac_loss
        return new_z1_list

    def power_method(self, f0, z0, n_iters=350):
        """ Computes the largest eigenvalues and corresponding eigenvectors using the power method"""
        evector = torch.randn_like(z0)
        bsz = evector.shape[0]
        for i in range(n_iters):
            vTJ = torch.autograd.grad(f0, z0, evector, retain_graph=(i < n_iters-1), create_graph=False)[0]
            evalue = (vTJ * evector).reshape(bsz, -1).sum(1, keepdim=True) / (evector * evector).reshape(bsz, -1).sum(1, keepdim=True)
            evector = (vTJ.reshape(bsz, -1) / vTJ.reshape(bsz, -1).norm(dim=1, keepdim=True)).reshape_as(z0)
        return (evector, torch.abs(evalue))

    def jac_loss_estimate(self, f0, z0, create_graph=True):
        """ Computes the jacobian regularization using the Hutchinson estimator"""
        bsz = z0.shape[0]
        vecs = 2    # z0 already has a batch dimension, which is stochastic!
        result = 0
        for i in range(vecs):
            v = torch.randn(*z0.shape).to(z0)
            vJ = torch.autograd.grad(f0, z0, v, retain_graph=True, create_graph=create_graph)[0]
            result += vJ.norm()**2
        return result / vecs / np.prod(z0.shape)

    def g(self, z, u, targ, fz, dim=(0,1)):
        """ 
        Computes the Cost -CE(y,targ)
        and also computes -CE(f(x, z),targ)
        """ 
        y = self.output_layer(z, self.u_cutoffs, self.bwd_pass).view(self.bsz, -1)
        fx = self.output_layer(fz, self.u_cutoffs, self.bwd_pass).view(self.bsz, -1)
        
        recon_x_vec = -self.criterion(y, targ)
        recon_x = recon_x_vec.mean()
        yhat = y.argmax(dim=1)
        return recon_x, recon_x_vec, recon_x, recon_x, fx, (yhat == targ).float().mean()*100# x

    def diffunc(self, uz, mu, rho, return_all=False, vec=False, only_cost=False, only_fx=False, only_fx_cost=False):
        """
        Computes the constraint and the cost
        constraint = diff_dec = f(z, u) - z
        cost = -CE(y,targ)
        """
        z, u = uz[:, :-self.u_size], uz[:, -self.u_size:]
        kl_loss = 0

        diff_dec, fz = self.diff_dec(u, z)
        if only_fx_cost:
            recon_x, recon_fx, recon_x_sq, recon_fx_sq, difffx, diffx = self.g(z, u, self.targ, fz)
            return fz, recon_x*np.prod(difffx.shape), recon_x, recon_fx, diffx

        if only_fx:
            return fz

    def diff_dec(self, u, z):
        """
        Compute the constraints = f(z, u) - z for the classifier
        """

        z_list_prime = DEQFunc2d.vec2list(z, self.z_cutoffs)
        u_list_prime = self.get_ulist(u, self.u_cutoffs, self.bwd_pass)
        if self.bwd_pass:
            fz_list = self.func_copy(z_list_prime, u_list_prime)
        else:
            fz_list = self.func(z_list_prime, u_list_prime)
        fz = DEQFunc2d.list2vec(fz_list).squeeze(-1)
        return (fz - z).reshape(self.bsz, -1), fz



class MDEQIMAMLWrapper(DEQModule2d):
    def __init__(self, func, func_copy):
        """
        MDEQ JIIO wrapper for Meta Network: Calls functions to perform input optimization
        and backward fixed points for a MDEQ meta network. Also defines helper functions for the 
        optimization (e.g defining objectives, computing forward pass through MDEQ layer etc.)
        """        
        super(MDEQIMAMLWrapper, self).__init__(func, func_copy)

    def forward(self, z_tr_list, z_te_list, v, u_tr_list, u_te_list, targ, output_layer, bsz_tr, bsz_te, final_layer_copy, classifier_copy, **kwargs):
        """Call functions to perform input optimization and backward fixed points"""

        train_step = kwargs.get('train_step', -1)
        threshold = kwargs.get('threshold', 30)
        writer = kwargs.get('writer', None)
        proj_adam = kwargs.get('proj_adam',False)
        alpha_0 = kwargs.get('alpha_0', [0.04,0.01])
        mem = kwargs.get('mem', 10)
        traj_reg = kwargs.get('traj_reg', False)
        jiio_thres = kwargs.get('jiio_thres', 100)
        if proj_adam:
            jiio_thres = 20
        self.output_layer = output_layer
        self.bwd_pass = False

        self.final_layer_copy = final_layer_copy
        self.classifier_copy = classifier_copy
        if targ.dtype == torch.float32:
            self.criterion = lambda output, target, dim: ((output - target)**2).mean(dim = dim)
        else:
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            self.criterion = lambda output, target, dim: criterion(output, target).mean() if len(dim)==2 else criterion(output, target)

        self.z_cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z_tr_list]
        self.u_cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in u_tr_list]

        u_tr_list = [u_tr_i.detach() for u_tr_i in u_tr_list]
        u_tr = DEQFunc2dMeta.list2vec(u_tr_list).squeeze(-1)
        u_te = DEQFunc2dMeta.list2vec(u_te_list).squeeze(-1)

        self.z_tr_list = z_tr_list
        self.z_te_list = z_te_list
        self.num_tasks, self.v_size = v.shape
        self.bsz_tr = bsz_tr
        self.bsz_te = bsz_te

        self.targ = targ

        # Call function to perform input optimization (optimizing v)
        new_z_tr, new_v, mu_final, z_tr_traj, v_tr_traj= DEQFunc2dMeta.apply(self.diffunc, z_tr_list, u_tr, v, True, proj_adam, mem, alpha_0, jiio_thres, train_step, writer, bsz_tr)

        new_vc = new_v.clone().detach()
        funcuv = lambda z, u, *args: self.funcuv(z, new_vc, u, *args)

        # Call function to perform z_te fixed points : z_te* = f(z_te*, new_v, u_te) given new_v
        new_z_te, new_v, _, _, _ = DEQFunc2dMeta.apply(funcuv, z_te_list, u_te_list, new_v, False, proj_adam, mem, alpha_0, threshold, train_step, writer, bsz_te)
        new_z_tr = Variable(new_z_tr, requires_grad=True)
        new_z_tr_copy = new_z_tr.squeeze(-1)
        if len(new_z_tr_copy.shape)==1:
            new_z_tr_copy = new_z_tr_copy.unsqueeze(-1)
            new_z_tr = new_z_tr.unsqueeze(-1)
        new_z_tr_list = DEQFunc2dMeta.vec2list(new_z_tr_copy, self.z_cutoffs)
        new_z_te = Variable(new_z_te, requires_grad=True)
        new_z_te_copy = new_z_te.squeeze(-1)
        new_z_te_list = DEQFunc2dMeta.vec2list(new_z_te_copy, self.z_cutoffs)

        new_v_tr_jc = v_tr_traj.clone().detach()
        new_z_tr_jc = Variable(z_tr_traj, requires_grad=True)
        new_z_tr_jc_list = DEQFunc2dMeta.vec2list(new_z_tr_jc, self.z_cutoffs)
        jac_loss = torch.tensor([0.,]).cuda()

        if self.training:
            new_z_tr_jc1 = DEQFunc2dMeta.list2vec(DEQFunc2dMeta.f(self.func, new_z_tr_jc_list, u_tr_list, new_v_tr_jc, threshold, train_step))
            jac_loss = self.jac_loss_estimate(new_z_tr_jc1.squeeze(-1), new_z_tr_jc)

            self.bwd_pass = True

            # Compute Jacobian regularization
            new_z_te = DEQFunc2dMeta.list2vec(DEQFunc2dMeta.f(self.func, new_z_te_list, u_te_list, new_v, threshold, train_step))
            jac_loss += self.jac_loss_estimate(new_z_te.squeeze(-1), new_z_te_copy)
            new_z_te = self.Backward.apply(self.func_copy, new_z_te, u_te_list, mu_final, 'meta', new_v, np.prod(targ.shape), threshold, train_step, writer)
            new_z_te_list = DEQFunc2dMeta.vec2list(new_z_te, self.z_cutoffs)
        self.new_z_te = new_z_te
        self.jac_loss = jac_loss
        self.new_v = new_v
        return new_z_te

    def power_method(self, f0, z0, n_iters=350):
        """ Computes the largest eigenvalues and corresponding eigenvectors using the power method"""
        evector = torch.randn_like(z0)
        bsz = evector.shape[0]
        for i in range(n_iters):
            vTJ = torch.autograd.grad(f0, z0, evector, retain_graph=(i < n_iters-1), create_graph=False)[0]
            evalue = (vTJ * evector).reshape(bsz, -1).sum(1, keepdim=True) / (evector * evector).reshape(bsz, -1).sum(1, keepdim=True)
            evector = (vTJ.reshape(bsz, -1) / vTJ.reshape(bsz, -1).norm(dim=1, keepdim=True)).reshape_as(z0)
        return (evector, torch.abs(evalue))

    def jac_loss_estimate(self, f0, z0, retain_graph=True, create_graph=True):
        """ Computes the jacobian regularization using the Hutchinson estimator"""
        bsz = z0.shape[0]
        vecs = 2    # z0 already has a batch dimension, which is stochastic!
        result = 0
        for i in range(vecs):
            v = torch.randn(*z0.shape).to(z0)
            vJ = torch.autograd.grad(f0, z0, v, retain_graph=True, create_graph=create_graph)[0]
            result += vJ.norm()**2
        return result / vecs / np.prod(z0.shape)

    def F(self, z, vi, mu, u):
        """ Computing the next iterates in the "augmented" DEQ """

        z = z.view(self.num_tasks, -1)
        mu = mu.view(self.num_tasks, -1)
        with torch.enable_grad():
            vi = Variable(vi, requires_grad = True).to(vi.device)
            z = Variable(z, requires_grad = True).to(z.device)
            mu = Variable(mu, requires_grad = True).to(mu.device)
            z0 = torch.cat([z, vi, mu], dim=1)
            z, vi, mu = z0[:, :z.shape[1]], z0[:, z.shape[1]:z.shape[1] + vi.shape[1]], z0[:, -mu.shape[1]:]
            v_exp = vi.repeat_interleave(self.bsz_tr, 0)
            z_exp = z.reshape(self.num_tasks*self.bsz_tr, -1)
            mu_exp = mu.reshape(self.num_tasks*self.bsz_tr, -1)
            alpha = [0.04,0.6,0.8]
            z_out, recon_x, recon_xp, recon_fxp, diff = self.diffunc(torch.cat([z_exp, v_exp], dim=1), u, 0, 0, only_fx_cost=True)
            mu_out, v_out = autograd.grad([z_out, recon_x], [z, vi], grad_outputs=[mu_exp, None], retain_graph=True, create_graph=True)
            z_out = z_out.reshape(z.shape)
            mu_temp = mu + alpha[1] * (mu_out - mu)
            v_temp = vi - alpha[0] * v_out
            z_temp = z + alpha[2] * (z_out - z)
        return torch.cat([z_temp, v_temp, mu_temp], dim=1), z0

    def funcuv(self, z, v, u, *args):
        v = v.repeat_interleave(self.bsz_te,0)
        return self.func(z, u, v, *args)

    def g(self, v, z, targ, fx, dim=(0,1)):
        """ 
        Computes the Cost CE(y_tr, targ) + lambda * v**2
        and also computes -CE(f(u_tr, z_tr, v), targ)
        """ 
        
        z = self.output_layer(z, self.z_cutoffs, self.bwd_pass)
        fx_o = self.output_layer(fx, self.z_cutoffs, self.bwd_pass)
        
        loss = self.criterion(z, targ, dim)
        coeff = 1
        recon_x = loss + coeff*(v**2).mean(dim=dim)
        recon_fx = self.criterion(fx_o, targ, dim) + coeff*(v**2).mean(dim=dim)

         
        if torch.isnan(recon_x.norm()):
            ipdb.set_trace()
        return recon_x, recon_fx, fx_o, z, loss

    def diffunc(self, zv, u, mu, rho, return_all=False, vec=False, only_cost=False, only_fx=False, only_fx_cost=False, lindiff=False):
        """
        Computes the constraint and the cost
        constraint = diff_dec = f(z_tr, u_tr, v) - z_tr
        cost = CE(y_tr, targ) + lambda * v**2
        """
        z, v = zv[:, :-self.v_size], zv[:, -self.v_size:]
        kl_loss = 0
        if only_cost:
            z = self.output_layer(z, v, self.z_cutoffs, self.bwd_pass).view(self.bsz, -1)
            recon_x = self.criterion(z, v, self.targ, (0,1))
            return recon_x

        if lindiff:
            diff_dec, fx = self.lin_diff(z,v, u)
        else:
            diff_dec, fx = self.diff_dec(z,v, u)
        
        if only_fx_cost:
            recon_x, recon_fx, difffx, diffx, loss = self.g(v, z, self.targ, fx, dim=(1,))
            return fx, recon_x.mean()*np.prod(difffx.shape), recon_x, recon_fx, diff_dec, loss
        if only_fx:
            return fx, diff_dec
        
        recon_x, recon_fx, difffx, diffx = self.g(v, z, self.targ, fx)
        lagrangian = (recon_x*np.prod(difffx.shape)*1 + torch.sum(mu*diff))
        if return_all:
            return diff, lagrangian, recon_x_sq, recon_fx_sq
        return diff, lagrangian

    def diff_dec(self, z, v, u):
        """
        Compute the constraints = f(z, u, v) - z for the classifier
        """

        z_list_prime = DEQFunc2dMeta.vec2list(z, self.z_cutoffs)
        u_list = DEQFunc2dMeta.vec2list(u, self.u_cutoffs)
        if self.bwd_pass:
            fx_list = self.func_copy(z_list_prime, u_list, v)
        else:
            fx_list = self.func(z_list_prime, u_list, v)
        fx = DEQFunc2dMeta.list2vec(fx_list).squeeze(-1)

        return (fx - z).reshape(z.shape[0], -1), fx#, fx_det

    def lin_diff(self, z, v, u):
        """
        Compute linearized constraints (as in Gauss Newton method) : We use forward mode autodiff to compute the linearization.
        """
        with torch.enable_grad():
            v0 = v.clone().detach()
            z0 = z.clone().detach()
            v_dim = v.shape[1]
            z_dim = z.shape[1]
            vz0 = torch.cat([v0, z0], dim=1).requires_grad_(True)
            diff0, fx0 = self.diff_dec(vz0[:,v_dim:], vz0[:,:v_dim], u)
            vz = torch.cat([v, z], dim=1)
            grad_outputs = torch.zeros_like(diff0, requires_grad=True)
            grad_inputs = torch.autograd.grad(diff0, vz0, grad_outputs, create_graph=True)
            diff_grad0 = torch.autograd.grad(grad_inputs, grad_outputs, vz - vz0, create_graph=True)[0]
            diff_lin = diff0 + diff_grad0
        return diff_lin, fx0
