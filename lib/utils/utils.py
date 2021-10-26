# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np


class FullModel(nn.Module):
    """
    Distribute the loss on multi-gpu to reduce 
    the memory cost in the main gpu.
    You can check the following discussion.
    https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
    """
    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, inputs, labels, train_step=-1, **kwargs):
        outputs = self.model(inputs, train_step=train_step)
        loss = self.loss(outputs, labels)
        return torch.unsqueeze(loss,0), outputs

def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
    

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name / cfg.COMMENT

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + cfg.COMMENT + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            #model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            #model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            #model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    return lr



from sklearn import mixture
# from sklearn.neighbors.kde import KernelDensity
import numpy as np
import pickle
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
# import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mse
# import ipdb
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
def get_sampler(latent_dim):
    def sample_z(args):
        mu, log_sigma = args
        eps = K.random_normal(shape=(K.shape(mu)[0], latent_dim), mean=0., stddev=1.)
        return mu + K.exp(log_sigma) * eps
    return sample_z


def loss_kl_divergence(mu, log_sigma):
    def kl_loss(y_true, y_pred):
        # Don't panic about the - sign. It has been pushed though into the barcket
        kl = 0.5 * K.sum(K.exp(2*log_sigma) + K.square(mu) - 1. - 2*log_sigma, axis=1)
        return kl

    return kl_loss


def total_loss(mu, log_sigma, kl_weight=1, recon_loss_func=None):
    def _vae_loss(y_true, y_pred):
        """ Calculate loss = reconstruction loss + KL loss for eatch data in minibatch """
        # E[log P(X|z)]
        if recon_loss_func is None:
            recon = mse(y_pred, y_true)
        else:
            recon = recon_loss_func(y_true, y_pred)

        # D_KL(Q(z|X) || P(z|X)); calculate in closed from as both dist. are Gaussian
        kl = loss_kl_divergence(mu, log_sigma)(0, 0)

        return K.sum(recon, axis=list(range(1, recon.shape.ndims))) + kl_weight * kl

    return _vae_loss

def get_vae(encoder, decoder, embeding_loss_weight, layer_for_z_sigma, recon_loss_func, constant_sigma):

    last_but_one_encoder_layer_output = encoder.get_layer(index=-2).output
    with tf.name_scope('encoder'):
        # log_sigma = _Dense(bottleneck_size, activation='tanh')(last_but_one_encoder_layer_output)
        e_in = encoder.inputs
        if constant_sigma is None:
            log_sigma = layer_for_z_sigma(last_but_one_encoder_layer_output)
            log_sigma = Lambda(lambda x: 5 * x, name='z_sigma')(log_sigma)
            encoder = Model(inputs=e_in, outputs=encoder.outputs + [log_sigma], name='std_vae_encoder_model')
        else:
            # Very nasty hack. Takes an input but always returns the same constant value!
            log_sigma = Lambda(lambda x: K.log(constant_sigma))(last_but_one_encoder_layer_output)

    with tf.name_scope('full_VAE'):
        mu = encoder.outputs[0]
        bottleneck_size = K.int_shape(encoder.outputs[0])[-1]
        z = Lambda(get_sampler(bottleneck_size))([mu, log_sigma])
        vae_out = decoder(z)
        vae = Model(inputs=e_in, outputs=vae_out, name='vae')
        vae.compile(optimizer=Adam(lr=1e-4), loss=total_loss(mu, log_sigma,
                                                                            kl_weight=embeding_loss_weight,
                                                                            recon_loss_func=recon_loss_func),
                    metrics=[loss_kl_divergence(mu, log_sigma), 'mse'])
        vae.summary()

    return encoder, decoder, vae


def print_memory():
        print(0, torch.cuda.max_memory_allocated(0)/(1024*1024*1024), torch.cuda.memory_reserved(0)/(1024*1024*1024), torch.cuda.memory_allocated(0)/(1024*1024*1024), torch.cuda.max_memory_reserved(0)/(1024*1024*1024))#, torch.cuda.memory_allocated(0)/(1024*1024*1024))
        print(1, torch.cuda.max_memory_allocated(1)/(1024*1024*1024), torch.cuda.memory_reserved(1)/(1024*1024*1024), torch.cuda.memory_allocated(1)/(1024*1024*1024), torch.cuda.max_memory_reserved(1)/(1024*1024*1024))#, torch.cuda.memory_allocated(0)/(1024*1024*1024))
        print(2, torch.cuda.max_memory_allocated(2)/(1024*1024*1024), torch.cuda.memory_reserved(2)/(1024*1024*1024), torch.cuda.memory_allocated(2)/(1024*1024*1024), torch.cuda.max_memory_reserved(2)/(1024*1024*1024))#, torch.cuda.memory_allocated(0)/(1024*1024*1024))
        print(3, torch.cuda.max_memory_allocated(3)/(1024*1024*1024), torch.cuda.memory_reserved(3)/(1024*1024*1024), torch.cuda.memory_allocated(3)/(1024*1024*1024), torch.cuda.max_memory_reserved(3)/(1024*1024*1024))#, torch.cuda.memory_allocated(0)/(1024*1024*1024))

class VaeModelWrapper:
    def __init__(self, input_shape, latent_space_dim, have_2nd_density_est, log_dir, sec_stg_beta):
        self.log_dir = log_dir
        self.latent_space_dim = latent_space_dim
        self.have_2nd_density_est = have_2nd_density_est
        self.sec_stg_beta = sec_stg_beta
        with tf.name_scope('Encoder'):
            e_in = Input(shape=input_shape)
            x = Dense(2048, activation='relu')(e_in)
            x = Dense(2048, activation='relu')(x)
            x = Dense(2048, activation='relu')(x)
            # x = Dense(2048, activation='relu')(x)
            z = Dense(latent_space_dim, activation='linear')(x)
            encoder = Model(inputs=e_in, outputs=z)

            layer_for_z_sigma = Dense(latent_space_dim, activation='tanh')

        with tf.name_scope('Decoder'):
            d_in = Input(shape=(latent_space_dim, ))
            x = Dense(2048, activation='relu')(d_in)
            x = Dense(2048, activation='relu')(x)
            x = Dense(2048, activation='relu')(x)
            # x = Dense(2048, activation='relu')(x)
            d_out = Dense(input_shape[0], activation='linear')(x)
            decoder = Model(inputs=d_in, outputs=d_out)

        self.encoder, self.decoder, self.auto_encoder = get_vae(
            encoder, decoder,
            embeding_loss_weight=self.sec_stg_beta,
            layer_for_z_sigma=layer_for_z_sigma,
            recon_loss_func=mean_squared_error,
            constant_sigma=None)

    def load(self, file_name):
        self.auto_encoder.load_weights(file_name)
        normalization = np.load(file_name+"_normalization.npz")
        self.data_std = normalization["data_std"]
        self.data_mean = normalization["data_mean"]
        if self.have_2nd_density_est:
            self.u_est = DensityEstimator(training_set=None, method_name="GMM_20", n_components=None,
                                          log_dir=self.log_dir)
            with open(file_name + '_2nd_density_est', 'rb') as f:
                self.u_est.model = pickle.load(f)

    def reconstruct(self, input_batch):
        inp_batch_normalized = (input_batch[:] - self.data_mean)/self.data_std
        inp_batch_recon = self.auto_encoder.predict(inp_batch_normalized)
        inp_batch_recon_dnorm = inp_batch_recon*self.data_std + self.data_mean
        return inp_batch_recon_dnorm

    def save(self, file_name):
        self.auto_encoder.save_weights(file_name)
        np.savez(file_name+"_normalization.npz", data_std=self.data_std, data_mean=self.data_mean)
        if self.have_2nd_density_est:
            with open(file_name+'_2nd_density_est', 'wb') as f:
                pickle.dump(self.u_est.model, f)

    def fit(self, training_data, de_2nd_name=None):
        self.data_mean = np.mean(training_data, axis=0)
        self.data_std = np.std(training_data, axis=0)
        print('data_mean + ' + str(self.data_mean))
        print('data_std + ' + str(self.data_std))

        self.training_data_normalized = ((training_data[:] - self.data_mean)/self.data_std)

        # callbacks
        cbs = []
        reduce_on_pl_cb = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1,
                                                            mode='auto',  min_delta=0.0001, cooldown=0, min_lr=0)
        cbs.append(reduce_on_pl_cb)
        # ipdb.set_trace()
        self.auto_encoder.fit(self.training_data_normalized, self.training_data_normalized, batch_size=128, epochs=20,
                              validation_split=0.1, verbose=1, callbacks=cbs)

        if self.have_2nd_density_est:
            u_train = self.encoder.predict(self.training_data_normalized)[0]
            self.u_est = DensityEstimator(training_set=u_train, method_name=de_2nd_name, n_components=None,
                                          log_dir=self.log_dir)
            self.u_est.fitorload()

    def sample(self, n_samples):
        if self.have_2nd_density_est:
            u_s = self.u_est.model.sample(n_samples)[0]
        else:
            u_s = np.random.normal(0, 1, size=(n_samples, self.latent_space_dim))

        z_s = self.decoder.predict(u_s, batch_size=200)
        z_s = z_s*self.data_std + self.data_mean
        return z_s, None

    def fit_2nd_density_est(self, training_data, de_2nd_name):
        # self.data_mean = np.mean(training_data, axis=0)
        # self.data_std = np.std(training_data, axis=0)
        print('data_mean + ' + str(self.data_mean))
        print('data_std + ' + str(self.data_std))

        self.training_data_normalized = ((training_data[:] - self.data_mean)/self.data_std)
        u_train = self.encoder.predict(self.training_data_normalized)[0]
        print("prediction finished")
        self.u_est = DensityEstimator(training_set=u_train, method_name=de_2nd_name, n_components=None,
                                      log_dir=self.log_dir)
        self.u_est.fitorload()

    def save_second(self, file_name):
        with open(file_name+'_2nd_density_est', 'wb') as f:
            pickle.dump(self.u_est.model, f)

class DensityEstimator:
    def __init__(self, training_set, method_name, n_components=None, log_dir=None, second_stage_beta=0.0005):
        self.log_dir = log_dir
        self.training_set = training_set
        self.fitting_done = False
        self.method_name = method_name
        self.second_density_mdl = None
        self.skip_fitting_and_sampling = False
        if method_name == "GMM_Dirichlet":
            self.model = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type='full',
                                                         weight_concentration_prior=1.0/n_components)
        elif method_name == "GMM":
            self.model = mixture.GaussianMixture(n_components=n_components, covariance_type='full', max_iter=2000,
                                                 verbose=2, tol=1e-3)
        elif method_name == "GMM_1":
            self.model = mixture.GaussianMixture(n_components=1, covariance_type='full', max_iter=2000,
                                                 verbose=2, tol=1e-3)
        elif method_name == "GMM_10":
            self.model = mixture.GaussianMixture(n_components=10, covariance_type='full', max_iter=2000,
                                                 verbose=2, tol=1e-3)
        elif method_name == "GMM_20":
            self.model = mixture.GaussianMixture(n_components=20, covariance_type='full', max_iter=2000,
                                                 verbose=2, tol=1e-3)
        elif method_name == "GMM_100":
            self.model = mixture.GaussianMixture(n_components=100, covariance_type='full', max_iter=2000,
                                                 verbose=2, tol=1e-3)
        elif method_name == "GMM_200":
            self.model = mixture.GaussianMixture(n_components=200, covariance_type='full', max_iter=2000,
                                                 verbose=2, tol=1e-3)

        elif method_name.find("aux_vae") >= 0:
            have_2nd_density_est = False
            if method_name[8:] != "":
                self.second_density_mdl = method_name[8:]
                have_2nd_density_est = True
            self.model = VaeModelWrapper(input_shape=(training_set.shape[-1], ),
                                         latent_space_dim=training_set.shape[-1]*3,
                                         have_2nd_density_est=have_2nd_density_est,
                                         log_dir=self.log_dir, sec_stg_beta=second_stage_beta)

        elif method_name == "given_zs":
            files = os.listdir(log_dir)
            for z_smpls in files:
                if z_smpls.endswith('.npy'):
                    break
            self.z_smps = np.load(os.path.join(log_dir, z_smpls))
            self.skip_fitting_and_sampling = True

        elif method_name.upper() == "KDE":
            self.model = KernelDensity(kernel='gaussian', bandwidth=0.425)
            # self.model = KernelDensity(kernel='tophat', bandwidth=15)
        else:
            raise NotImplementedError("Method specified : " + str(method_name) + " doesn't have an implementation yet.")

    def fitorload(self, file_name=None):
        if not self.skip_fitting_and_sampling:
            if file_name is None:
                self.model.fit(self.training_set, self.second_density_mdl)
            elif self.method_name.upper().find("VAE") >= 0:
                self.model.load(file_name)
            else:
                with open(file_name, 'rb') as f:
                    self.model = pickle.load(f)

        self.fitting_done = True

    def fit2nd(self,de_2nd_name):
        self.model.fit_2nd_density_est(self.training_set, de_2nd_name)

    def score(self, X, y=None):
        if self.method_name.upper().find("AUX_VAE") >= 0 or self.skip_fitting_and_sampling:
            raise NotImplementedError("Log likelihood evaluation for VAE is difficult. or skipped")
        else:
            return self.model.score(X, y)

    def save(self, file_name):
        if not self.skip_fitting_and_sampling:
            if self.method_name.find('vae') >= 0:
                self.model.save(file_name)
            else:
                with open(file_name, 'wb') as f:
                    pickle.dump(self.model, f)

    def save_second(self, file_name):
        self.model.save_second(file_name)


    def reconstruct(self, input_batch):
        if self.method_name.upper().find("AUX_VAE") < 0:
            raise ValueError("Non autoencoder style density estimator: " + self.method_name)
        return self.model.reconstruct(input_batch)

    def get_samples(self, n_samples):
        if not self.skip_fitting_and_sampling:
            if not self.fitting_done:
                self.fitorload()
            scrmb_idx = np.array(range(n_samples))
            np.random.shuffle(scrmb_idx)
            if self.log_dir is not None:
                pickle_path = os.path.join(self.log_dir, self.method_name+'_mdl.pkl')
                with open(pickle_path, 'wb') as f:
                    pickle.dump(self.model, f)
            if self.method_name.upper() == "GMM_DIRICHLET" or self.method_name.upper() == "AUX_VAE" \
                    or self.method_name.upper() == "GMM" or self.method_name.upper() == "GMM_1" \
                    or self.method_name.upper() == "GMM_10" or self.method_name.upper() == "GMM_20" \
                    or self.method_name.upper() == "GMM_100" or self.method_name.upper() == "GMM_200"\
                    or self.method_name.upper().find("AUX_VAE") >= 0:
                return self.model.sample(n_samples)[0][scrmb_idx, :]
            else:
                return np.random.shuffle(self.model.sample(n_samples))[scrmb_idx, :]
        else:
            return self.z_smps

def fit_density_model(latents_file=None, latents=None):
    if latents is None:
        latents = np.load(latents_file)
    latents = latents[:100000]
    sampler = DensityEstimator(training_set=latents, method_name='aux_vae_GMM_20', n_components=20)
    sampler.fitorload()
    smpls = sampler.get_samples(n_samples=20000)
    estimated_covar = np.cov(smpls.T)
    if latents_file is not None:
        save_folder = '/'.join(latents_file.split('/')[:-1])
        sampler.save(save_folder + '/latents_vaesampler_x4_3l2048')
        np.save(save_folder + '/vaesampler_x3_3l2048_samples_gmm20.npy', smpls)
    return smpls, sampler
