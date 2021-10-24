#!/usr/bin/env python
# coding: utf-8

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os
import torch.nn.functional as F

from torch.nn import Linear
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
import scipy
import random
import json
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler,WeightedRandomSampler

from scipy.signal import convolve2d
from generative_model import realnvpfc_model
import argparse
import imageio



class KernelNetwork(torch.nn.Module):
    def __init__(self,num_layers = 3):
        super(KernelNetwork, self).__init__()
        layers = []
        self.num_layers = num_layers
        for i in range(num_layers):
            layers.append(torch.nn.Conv2d(1, 1, 3, padding=1, bias=False, padding_mode="zeros"))
        self.layers = layers
        self.net = nn.Sequential(*layers)

    def load(self,filepath,device):
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)

    def forward(self, x): 
        for i in range(self.num_layers):
            x = self.net[i](x)
        return x
    
# class KNetwork(torch.nn.Module):
#     def __init__(self,num_layers = 3, layer_size = 3):
#         super(KNetwork, self).__init__()
#         layers = []
#         self.num_layers = num_layers
#         for i in range(num_layers):
#             layers.append(torch.nn.Parameter(torch.randn(1, 1, layer_size, layer_size)))
#         self.layers = layers
#         self.padding = (layer_size - 1)//2

#     def load(self,filepath,device):
#         checkpoint = torch.load(filepath, map_location=device)
#         self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
#     def generatekernel(self):
#         for i in range(self.num_layers-1):
#             if i == 0:
#                 ker = F.conv2d(self.layers[i], self.layers[i+1], padding=self.padding )
#             else:
#                 ker = F.conv2d(ker, self.layers[i+1], padding = self.padding)
# #             print(ker.shape)
#         ker_norm = ker/torch.sum(ker)
#         return ker_norm

#     def forward(self, x):
#         k = self.generatekernel()
#         out =  F.conv2d(x, k, padding=self.padding )
# #         print(out.shape)
#         return x
    
class KNetwork(torch.nn.Module):
    def __init__(self,num_layers = 3, layer_size = 3, softplus=False, hardk = False, beta = 1, padding_mode = "reflect"):
        super(KNetwork, self).__init__()
        layers = []
        self.num_layers = num_layers
#         for i in range(num_layers):
#             layers.append(torch.nn.Parameter(torch.randn(1, 1, layer_size, layer_size)))
#         self.layers = layers

#         gauss = Tensor([[[[[0, 0, 0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0, 0, 0],
#                       [0, 0, 0, 0.16, 0, 0, 0], 
#                       [0, 0, 0.16, 0.3, 0.16, 0, 0], 
#                       [0, 0, 0, 0.16, 0, 0, 0],
#                      [0, 0, 0, 0, 0, 0, 0],
#                      [0, 0, 0, 0, 0, 0, 0]]]]])*5
#         gauss_full = torch.zeros([1, 1, 1, layer_size, layer_size])
#         c = int((layer_size - 7)/2)
#         gauss_full[:, :, :, c:c+7, c:c+7] = gauss
#         gauss_cat = torch.cat(num_layers*[gauss_full])

        gauss = Tensor(makeGaussian(layer_size, fwhm = layer_size//2, peak = 0.03, center=None)[np.newaxis, np.newaxis, np.newaxis, :])
        gauss_cat = torch.cat(num_layers*[gauss])
        
        self.layers = torch.nn.Parameter(gauss_cat, requires_grad = True)
#         self.layers = torch.nn.Parameter(torch.randn(num_layers, 1, 1, layer_size, layer_size), requires_grad = True)
        self.padding = (layer_size - 1)//2
        self.softplus = softplus
        self.hardk = hardk
        self.padding_mode = padding_mode
        self.beta = beta

    def load(self,filepath,device):
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'], strict=True)
        
    def generatekernel(self):
        for i in range(self.num_layers-1):
            if i == 0:
                ker = F.conv2d(self.layers[i], self.layers[i+1], padding=self.padding)
            else:
                ker = F.conv2d(ker, self.layers[i+1], padding = self.padding)
        if self.softplus == True:
            ker_soft = torch.nn.Softplus(beta=self.beta)(ker)
            ker = ker_soft
        if self.hardk == True:
            ker_norm = ker/torch.sum(ker)
            return ker_norm
        else:
            return ker

    def forward(self, x):
        k = self.generatekernel()
        out =  F.conv2d(x, k, padding=self.padding)
        return out



######################################################################################################################

        
#                            EM
    

######################################################################################################################

# def GForward(z_sample, img_generator, nsrc, d, logscale_factor, eiko=False, xtrue=None):
#     if eiko==False:
#         x_samples_transformed, logdet = img_generator.reverse(z_sample)
#         img = torch.sigmoid(x_samples_transformed)
#         det_sigmoid = torch.sum(-x_samples_transformed -  2*torch.nn.Softplus()(-x_samples_transformed), -1)

#         logdet = logdet + det_sigmoid
#         img = img.reshape(-1, nsrc, d)
#     else:
#         img=xtrue
#         logdet = 0
#     return img, logdet

# def FForward(x, Xrec, kernel_network, sigma, v, device):
#     y = kernel_network(Xsrc=x, Xrec=Xrec, v=v) 
#     noise = torch.randn(y.shape)*sigma
#     y += noise.to(device)
#     return y

def GForward(z_sample, img_generator, npix, logscale_factor):
#     print(len(z_sample), z_sample[0].shape)
    img_samp, logdet = img_generator.reverse(z_sample)
    img_samp = img_samp.reshape((-1, npix, npix))
    
    # apply scale factor and sigmoid/softplus layer for positivity constraint
    logscale_factor_value = logscale_factor.forward()
    scale_factor = torch.exp(logscale_factor_value)
    img = torch.nn.Softplus()(img_samp) * scale_factor
    det_softplus = torch.sum(img_samp - torch.nn.Softplus()(img_samp), (1, 2))
    det_scale = logscale_factor_value * npix * npix
    logdet = logdet + det_softplus + det_scale
    return img, logdet

def FForward(x, kernel_network, sigma, device, hard_k, kerl1):
#     if hard_k == True and softmax == False:
# #         print("Using hard constraint")
#         kl1 = kerl1(kernel_network)
# #         x /= kl1
#         x = x/kl1
    y = kernel_network(torch.unsqueeze(x, dim=1))
    y = torch.squeeze(y, dim=1)
    noise = torch.randn(y.shape)*sigma
    y += noise.to(device)
    return y


def EStep(z_sample, device, ytrue, img_generator, kernel_network, prior, logdet_weight, 
          px_weight, sigma, npix, logscale_factor, data_weight, hard_k, kerl1):    
    img, logdet = GForward(z_sample, img_generator, npix, logscale_factor)
    y = FForward(img, kernel_network, sigma, device, hard_k, kerl1)

    logqtheta = -logdet_weight*torch.mean(logdet)
    meas_err = data_weight*torch.mean(nn.MSELoss()(y, ytrue)) 
#     print(y.shape, ytrue.shape)
    prior_x = torch.mean(prior(img, px_weight))
    loss = logqtheta + prior_x + meas_err
    return loss, logqtheta, prior_x, meas_err


def MStep(z_sample, x_sample, npix, device, ytrue, img_generator, kernel_network, phi_weight, 
          fwd_network, sigma, kernelweight, logscale_factor, ker_softl1, kerl1, hard_k, prior_phi, prior_phi_weight):
    
    img, logdet = GForward(z_sample, img_generator, npix, logscale_factor)
    y = FForward(img, kernel_network, sigma, device, hard_k, kerl1)
    y_x = FForward(x_sample, kernel_network, sigma, device, hard_k, kerl1)
    fwd = FForward(x_sample, fwd_network, sigma, device, hard_k, kerl1)
    pphi = phi_weight*nn.MSELoss()(y_x, fwd)
    kernel = kernel_network.generatekernel()
    
    prior = prior_phi_weight*prior_phi(kernel)
    soft_k = kernelweight*ker_softl1(kernel_network)
    meas_err = nn.MSELoss()(y, ytrue)
#     print(y_x.shape, fwd.shape, y.shape, ytrue.shape)
    loss =  pphi + meas_err + soft_k + prior
    return loss, pphi, meas_err, soft_k, prior

def generate_sample(model_form, btsize, npix, device, z_shapes):
    if model_form == 'realnvp':
        z_sample = torch.randn(btsize, npix*npix).to(device=device)
#                 z_sample = torch.randn(n_batch, npix*npix).to(device=device)
    elif model_form == 'glow':
        z_sample = []
        for z in z_shapes:
            z_new = torch.randn(btsize, *z)
            z_sample.append(z_new.to(device))
#             z_sample = torch.randn(args.btsize, npix*npix).to(device=args.device)
    return z_sample

#########################################################################################################

        
#                            Model Helpers
    

###########################################################################################################


    
def init_weights(m):
    if type(m) == nn.Linear:
#         torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.normal_(m.weight, mean=0.0, std=1e-4)
        m.bias.data.fill_(0.01)
        
def init_kernel_network(m):
    if type(m) == nn.Conv2d:
#         torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.normal_(m.weight, mean=0.0, std=1e-1)
#         m.bias.data.fill_(0)
#         m.bias.requires_grad = False

def init_kernel_gauss(m):
    gauss = Tensor([[[[0.01, 0.16, 0.01], [0.16, 0.3, 0.16], [0.01, 0.16, 0.01]]]])
    if type(m) == nn.Conv2d:
#         torch.nn.init.xavier_uniform_(m.weight)
        m.weight = nn.Parameter(gauss)
    
def init_kernel_gauss7(m):
#     gauss = Tensor([[[[0.01, 0.16, 0.01], [0.16, 0.3, 0.16], [0.01, 0.16, 0.01]]]])
    gauss = Tensor([[[[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0.01, 0.16, 0.01, 0, 0], 
                      [0, 0, 0.16, 0.3, 0.16, 0, 0], 
                      [0, 0, 0.01, 0.16, 0.01, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0]]]])
    if type(m) == nn.Conv2d:
#         torch.nn.init.xavier_uniform_(m.weight)
        m.weight = nn.Parameter(gauss)

######################################################################################################################

        
#                            DPI
    

######################################################################################################################



class Img_logscale(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, scale=1):
        super().__init__()
        log_scale = torch.Tensor(np.log(scale)*np.ones(1))
        self.log_scale = nn.Parameter(log_scale)

    def forward(self):
        return self.log_scale
    
    
def Loss_l1(y_pred):
    # image prior - sparsity loss
    return torch.mean(torch.abs(y_pred), (-1, -2))

def Loss_TSV(y_pred):
    # image prior - total squared variation loss
    return torch.mean((y_pred[:, 1::, :] - y_pred[:, 0:-1, :])**2, (-1, -2)) + torch.mean((y_pred[:, :, 1::] - y_pred[:, :, 0:-1])**2, (-1, -2))

def Loss_TV(y_pred):
    # image prior - total variation loss
    return torch.mean(torch.abs(y_pred[:, 1::, :] - y_pred[:, 0:-1, :]), (-1, -2)) + torch.mean(torch.abs(y_pred[:, :, 1::] - y_pred[:, :, 0:-1]), (-1, -2))




######################################################################################################################

        
#                            ADDITIONAL STUFF
    

######################################################################################################################

def makeGaussian(size, fwhm = 3, peak = 1, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    mat = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    mat /= np.max(mat)/peak
    return mat

def generate_image(imsize = [32, 32]):
    img = np.zeros(imsize)
    for i in range(100):
        x = np.random.randint(0, imsize, 2)
        size = np.random.randint(0, 5, 2)
        val = np.random.randint(0, 10)/10
        img[x[0]-size[0]:x[0]+size[0],x[1]-size[1]:x[1]+size[1] ] = val
    return img

def sample_x_deadleaf(n_batch, npix):
    x = np.zeros([n_batch, npix, npix])
    for i in range(n_batch):
        x[i, :] = generate_image(imsize = [npix, npix])
    return x

def create_kernel_torch(kernel_network, device, num_layers):
    delta = torch.zeros([1, 1, num_layers*2+1, num_layers*2+1])
    delta[0, 0, num_layers, num_layers] = 1
    learned_kernel=kernel_network(delta.to(device))
    return learned_kernel

def create_kernel_np(kernel_network):
    learned_kernel = np.squeeze(np.squeeze(kernel_network.net[0].weight.data.detach().cpu().numpy(), axis=0), axis=0)
    for i in range(2):
        arr = np.squeeze(np.squeeze(kernel_network.net[i+1].weight.data.detach().cpu().numpy(), axis=0), axis=0)
        learned_kernel = convolve2d(learned_kernel, arr, 
                                                 mode='full', boundary='fill', fillvalue=0)
    return learned_kernel

def generate_gif(DIR, name, epoch, subepoch, s, subs):
    images = []
    for i in range(0, epoch, s):
        for j in range(0, subepoch, subs):
            filename = '{}/{}{}_{}.png'.format(DIR, name, str(i).zfill(5),str(j).zfill(5))
            images.append(imageio.imread(filename))
    imageio.mimsave('{}/{}.gif'.format(DIR, name), images)
    return