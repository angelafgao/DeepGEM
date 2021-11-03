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
import eikonalfm
import random
import json
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler,WeightedRandomSampler

from scipy.interpolate import interp2d, griddata
from scipy.signal import convolve2d
from generative_model import realnvpfc_model
import argparse
import imageio

######################################################################################################################

        
"""                          Helper Functions for EM 
"""
    
######################################################################################################################

def GForward(z_sample, GNet, nsrc, d, logscale_factor, device, eiko=False, 
             sampled = False, std=1e-2, samplenoise = None, xtrue=None, use_dataparallel = False):
""" Forward pass of G network (sample the posterior network)
    
Arguments: 
    z_sample: random samples of latent space
    GNet: posterior estimation network
    nsrc: number of sources
    d: dimension of space
    logscale_factor: normalizing value
    device: network device
    eiko: M step only generating samples
    sampled: sample from prior
    std: std of prior
    samplenoise: add specific error to samples
    xtrue: true sources
    use_dataparallel

"""
    if eiko==False:
        if use_dataparallel==True:
            x_samples_transformed, logdet = GNet.module.reverse(z_sample)
        else:
            x_samples_transformed, logdet = GNet.reverse(z_sample)
        img = torch.sigmoid(x_samples_transformed)
        det_sigmoid = torch.sum(-x_samples_transformed -  2*torch.nn.Softplus()(-x_samples_transformed), -1)

        logdet = logdet + det_sigmoid
        img = img.reshape(-1, nsrc, d)
    else:
        if sampled == True:
            if samplenoise is None:
                img = std*torch.randn(xtrue.shape).to(device) + xtrue
                logdet = 0
            else:
                img = xtrue + samplenoise
                logdet = 0
        else:
            img=xtrue
            logdet = 0
    return img, logdet

def FForward(x, Xrec, FNet, sigma, v, device, fwdmodel=None, nopairs=False, vinvar = False, 
             velo_loss = False, retain_graph = True):
""" Forward pass through forward model
    
Arguments: 
    x: sources
    Xrec: recievers
    FNet: forward model network
    sigma: std of noise
    v: homogeneous velocity value
    device: device of model
    fwdmodel: forward model 
    nopairs: no pairs for generating travel times
    vinvar: vinvar network
    velo_loss: using velocity loss
    retain_graph: for L_T

"""
    if fwdmodel is None:
        y = FNet(Xsrc=x, Xrec=Xrec, v=v, nopairs = nopairs, velo=velo_loss, vinvar = vinvar, 
                           device=device, retain_graph=retain_graph) 
    else:
        y = FNet(idx = x)
    noise = torch.randn(y.shape)*sigma
    y += noise.to(device)
    return y

def EStep(z_sample, XrecIn, device, ytrue, GNet, FNet, prior, data_weight, 
          prior_weight, prior_sigma, data_sigma, velocity, nsrc, d, logscale_factor, logdet_weight, velo_loss, 
          use_dataparallel, reduction):
""" Single E Step 
    
Arguments: 
    z_sample: random samples from latent space
    XrecIn: receiver locations
    device: model device
    ytrue: true measurements
    GNet:  posterior network
    FNet: forward model
    prior: p(x)
    data_weight: 1/sigma**2 for measurement error sigma
    prior_weight: 1/sigma**2 for p(x)
    prior_sigma: for p(x)
    data_sigma: for measurement likelihood
    velocity: velocity model
    nsrc: number sources
    d: dimension of space
    logscale_factor: scaling term
    logdet_weight: weight entropy
    velo_loss: use velocity loss
    use_dataparallel
    reduction: mse vs sse

"""
    img, logdet = GForward(z_sample, GNet, nsrc, d, logscale_factor, device=device,
                           eiko=False, xtrue=None, use_dataparallel = use_dataparallel)
    y = FForward(img, XrecIn, FNet, data_sigma, velocity, device, velo_loss=velo_loss)

    logqtheta = -logdet_weight*torch.sum(logdet)
    meas_err = data_weight*torch.mean(nn.MSELoss(reduction=reduction)(y, ytrue)) #data weight = 1/sigma**2
    prior_x = torch.sum(prior(img))*prior_weight # prior gauss sum||x-x_mu||/sigma**2
    loss = logqtheta + prior_x + meas_err
    return loss, logqtheta, prior_x, meas_err



def MStep(z_sample, XrecIn, x_sample_src, x_sample_rec, device, ytrue, GNet, FNet, phi_weight, fwd_network,
          data_sigma, velocity, fwd_velocity, nsrc, d, logscale_factor, eiko, xtrue, fwdmodel, xidx, velo_loss, 
          invar_weight, invar_src, invar_rec, VNet, ttinvar, fwd_velocity_model, sampled,samplenoise,
          prior_x,use_dataparallel = False):
""" Single M Step
    
Arguments: 
    z_sample: random samples from latent space
    XrecIn: receiver locations
    x_sample_src: ____
    x_sample_rec: ____
    device: network device
    ytrue: measurements
    GNet: posterior network
    FNet: forward model:
    phi_weight: lambda_theta
    fwd_network: ____
    data_sigma: measurement error
    velocity: homogeneous velocity
    fwd_velocity: assumed homogeneous velocity value
    nsrc: number of sources
    d: dimension
    logscale_factor: scaling factor
    eiko: M step only
    xtrue: true source locations
    fwdmodel: ____
    xidx: ____
    velo_loss: loss on velocity error
    invar_weight: lambda_V
    invar_src: sources used for L_V L_T
    invar_rec: receivers used for L_V L_T
    VNet: ______
    ttinvar: lambda_T
    fwd_velocity_model: V used for L_theta
    sampled: sample from prior (used for first M step to initialize theta)
    samplenoise: noise added (used for first M step to initialize theta)
    prior_x: p(x) std

"""

    img, logdet = GForward(z_sample, GNet, nsrc, d, logscale_factor, device=device,
                           eiko=eiko, sampled=sampled, std= prior_x,
                           xtrue=xtrue, use_dataparallel=use_dataparallel, samplenoise=samplenoise)
    y = FForward(img, XrecIn, FNet, data_sigma, velocity, device, velo_loss=velo_loss)
    meas_err = nn.MSELoss()(y, ytrue)

    # Fwd model prior as velocity prior
    if phi_weight > 0:
        y_x = FForward(invar_src, invar_rec, FNet, data_sigma, velocity, device, vinvar = True, nopairs=True, velo_loss=True)
        
        x = np.linspace(0, 1, fwd_velocity_model.shape[0])
        X, Y = np.meshgrid(x, x)
        idx = np.zeros([(fwd_velocity_model.shape[0])**2, 2])
        idx[:, 0] = Y.flatten(); idx[:, 1] = X.flatten()
        fwd = griddata(idx, fwd_velocity_model.flatten(), invar_rec.detach().cpu().numpy().squeeze(0))

        fwdext = np.concatenate(invar_src.shape[1]*[fwd[np.newaxis, :]], axis=0)
        pphi = phi_weight*nn.MSELoss()(y_x, Tensor(fwdext).to(device))
    else:
        pphi = 0
        
#    Velocity Invariance Prior 2 src, n rec
    if invar_weight > 0:
        invar_src_in = invar_src[:, 0:2, :]
        
        y_x = FForward(invar_src_in, invar_rec, FNet, data_sigma, velocity, device, vinvar = True, nopairs=True, velo_loss=True)
        mse_invar = invar_weight*nn.MSELoss()(y_x[0, :], y_x[1, :])
    else:
        mse_invar = 0
        
        
#    Travelt Time Invariance Prior 2 src, n rec
    if ttinvar is not None: # travel time invariance loss
        invar_rec.requires_grad = True
        
        XsrcInSingle = torch.unsqueeze(invar_src[0,:], axis=0)
        XrecInSingle = torch.unsqueeze(invar_rec[0,:], axis=0)
        
        y1 = FForward(XsrcInSingle, XrecInSingle, 
                          FNet, data_sigma, velocity, device, velo_loss=False, retain_graph = True) # btsize x nsrc x nrec
        y1_nograd = y1.detach()

        y2 = FForward(XrecInSingle, XsrcInSingle, 
                          FNet, data_sigma, velocity, device, velo_loss=False, retain_graph = True) # btsize x nrec x nsrc
        y2_transposed =  torch.transpose(y2, -1, -2)
        y2_nograd = torch.transpose(y2.detach(), -1, -2)
       
        mse_ttinvar = ttinvar*nn.MSELoss()(y1, y2_transposed)
    else:
        mse_ttinvar = 0
        
    
    
    loss =  pphi + meas_err + mse_invar  + mse_ttinvar
    return loss, pphi, meas_err, mse_invar, mse_ttinvar


#########################################################################################################

        
#                            MAP Helper
    

###########################################################################################################


def MAPStep(z_sample, XrecIn, x_sample_src, x_sample_rec, device, ytrue, GNet, FNet, phi_weight, fwd_network,
          data_sigma, velocity, fwd_velocity, nsrc, d, logscale_factor, eiko, xtrue, fwdmodel, xidx, velo_loss, 
          invar_weight, invar_src, invar_rec, vnet_weight, ttinvar, fwd_velocity_model, sampled,samplenoise,
          prior_x, data_weight,reduction,prior,prior_weight,use_dataparallel = False):
""" single MAP_{x theta} step
    
Arguments: 
    z_sample: random samples from latent space
    XrecIn: receiver locations
    x_sample_src: ____
    x_sample_rec: ____
    device: network device
    ytrue: measurements
    GNet: posterior network
    FNet: forward model:
    phi_weight: lambda_theta
    fwd_network: ____
    data_sigma: measurement error
    velocity: homogeneous velocity
    fwd_velocity: assumed homogeneous velocity value
    nsrc: number of sources
    d: dimension
    logscale_factor: scaling factor
    eiko: M step only
    xtrue: true source locations
    fwdmodel: ____
    xidx: ____
    velo_loss: loss on velocity error
    invar_weight: lambda_V
    invar_src: sources used for L_V L_T
    invar_rec: receivers used for L_V L_T
    VNet: ______
    ttinvar: lambda_T
    fwd_velocity_model: V used for L_theta
    sampled: sample from prior (used for first M step to initialize theta)
    samplenoise: noise added (used for first M step to initialize theta)
    prior_x: p(x) std
    data_weight: weight on data likelihood for data_sigma
    reduction: mse vs sse
    prior: p(x)
    prior_weight: error on source locations

"""    
    
    img, logdet = GForward(z_sample, GNet, nsrc, d, logscale_factor, device=device,
                           eiko=False, xtrue=None, use_dataparallel = use_dataparallel)
    y = FForward(img, XrecIn, FNet, data_sigma, velocity, device, velo_loss=velo_loss)

    meas_err = data_weight*torch.mean(nn.MSELoss(reduction=reduction)(y, ytrue)) #data weight = 1/sigma**2
    
    prior_err = torch.sum(prior(img))*prior_weight # prior gauss sum||x-x_mu||/sigma**2


    # Fwd model prior as velocity prior
    if phi_weight > 0:
        y_x = FForward(invar_src, invar_rec, FNet, data_sigma, velocity, device, vinvar = True, nopairs=True, velo_loss=True)
        
        x = np.linspace(0, 1, fwd_velocity_model.shape[0])
        X, Y = np.meshgrid(x, x)
        idx = np.zeros([(fwd_velocity_model.shape[0])**2, 2])
        idx[:, 0] = Y.flatten(); idx[:, 1] = X.flatten()
        fwd = griddata(idx, fwd_velocity_model.flatten(), invar_rec.detach().cpu().numpy().squeeze(0))

        fwdext = np.concatenate(invar_src.shape[1]*[fwd[np.newaxis, :]], axis=0)
        pphi = phi_weight*nn.MSELoss()(y_x, Tensor(fwdext).to(device))
    else:
        pphi = 0
        
#    Velocity Invariance Prior 2 src, n rec
    if invar_weight > 0:
        invar_src_in = invar_src[:, 0:2, :]
        
        y_x = FForward(invar_src_in, invar_rec, FNet, data_sigma, velocity, device, vinvar = True, nopairs=True, velo_loss=True)
        mse_invar = invar_weight*nn.MSELoss()(y_x[0, :], y_x[1, :])
    else:
        mse_invar = 0
        
        
#    Travelt Time Invariance Prior 2 src, n rec
    if ttinvar is not None: # travel time invariance loss
        invar_rec.requires_grad = True
        
        XsrcInSingle = torch.unsqueeze(invar_src[0,:], axis=0)
        XrecInSingle = torch.unsqueeze(invar_rec[0,:], axis=0)
        
        y1 = FForward(XsrcInSingle, XrecInSingle, 
                          FNet, data_sigma, velocity, device, velo_loss=False, retain_graph = True) # btsize x nsrc x nrec
        y1_nograd = y1.detach()

        y2 = FForward(XrecInSingle, XsrcInSingle, 
                          FNet, data_sigma, velocity, device, velo_loss=False, retain_graph = True) # btsize x nrec x nsrc
        y2_transposed =  torch.transpose(y2, -1, -2)
        y2_nograd = torch.transpose(y2.detach(), -1, -2)
       
        mse_ttinvar = ttinvar*nn.MSELoss()(y1, y2_transposed)
    else:
        mse_ttinvar = 0
        
    
    
    loss =  pphi + meas_err + prior_err + mse_invar  + mse_ttinvar
    return loss, pphi, meas_err, prior_err, mse_invar, mse_ttinvar




#########################################################################################################

        
#                            Initialize Model
    

###########################################################################################################


    
def init_weights(m):
"""init model weights 
"""
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean=0.0, std=1e-1)
        m.bias.data.fill_(0.01)
        
        
def init_weights_eiko(m):
""" init weights of eikonet
"""    
    if type(m) == torch.nn.Linear:
        stdv = (1. / math.sqrt(m.weight.size(1))/1.)*2
        m.weight.data.uniform_(-stdv,stdv)
        if m.bias.data is not None:
            m.bias.data.uniform_(-stdv,stdv)
        else:
            m.weight.data.fill_(1.0)
        
def init_weights_eiko_sine(m):
""" init weights of eikonet with sine activation
"""    
    if type(m) == torch.nn.Linear:
        stdv = (1. / math.sqrt(m.weight.size(1))/1.)
        m.weight.data.uniform_(-stdv,stdv)
        if m.bias.data is not None:
            m.bias.data.uniform_(-stdv,stdv)
        else:
            m.weight.data.fill_(1.0)
            
def init_weights_eiko_ffreqs(m):
""" init weights of eikonet with fourier features positional encoding
"""    
    if type(m) == torch.nn.Linear:
        stdv = (1. / math.sqrt(m.weight.size(1))/1.)
        m.weight.data.uniform_(-stdv,stdv)
        if m.bias.data is not None:
            m.bias.data.uniform_(-stdv,stdv)
        else:
            m.weight.data.fill_(1.0)
            
            
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
"""image prior - sparsity loss"""
    return torch.mean(torch.abs(y_pred), (-1, -2))

def Loss_TSV(y_pred):
""" image prior - total squared variation loss """
    return torch.mean((y_pred[:, 1::, :] - y_pred[:, 0:-1, :])**2, (-1, -2)) + torch.mean((y_pred[:, :, 1::] - y_pred[:, :, 0:-1])**2, (-1, -2))

def Loss_TV(y_pred):
""" image prior - total variation loss """
    return torch.mean(torch.abs(y_pred[:, 1::, :] - y_pred[:, 0:-1, :]), (-1, -2)) + torch.mean(torch.abs(y_pred[:, :, 1::] - y_pred[:, :, 0:-1]), (-1, -2))



######################################################################################################################

        
#                           Plotting Functions
    

######################################################################################################################

def get_cmap(n, name='nipy_spectral'):
'''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

# def PlotVeloSlices(path, filename, plotRec, FNet, device, s, true_velocity_model, savefinal = False):
# """
    
# Arguments: 
#     path: output path
#     filename: output filename
#     plotRec:
#     FNet: forward model network
#     device:
#     s:
#     true_velocity_model
#     savefinal

# """
#     fig, ax = plt.subplots(2, 4, figsize=(14, 8))
#     for i in range(0, 4):
#         if pltRec == True:
#             src = plotRec[i, :]
#         else:
#             src = [0.25*i, 0.25*i]
#         V, TT = VeloRecon(FNet, device, num = s, src=src)
#         V = V.detach().cpu().numpy().reshape([s, s])
#         TT = TT.detach().cpu().numpy().reshape([s, s])
#         error_im = np.abs(V.transpose()-true_velocity_model)
#         err = np.sum(error_im)/51/51
        
#         if savefinal == True:
#             np.save("{}/Data/VFinalRecon{}_{}.npy".format(path, src[1], src[0]), V)

#         im=ax[0, i].imshow(V.transpose(), vmin=0, vmax=10)
#         fig.colorbar(im, ax=ax[0, i],fraction=0.046, pad=0.04)
#         ax[0, i].set_title("Velocity {} {} {:.3f}".format(int(50*src[1]), int(50*src[0]),err))
        
#         im=ax[1, i].imshow(np.abs(V.transpose()-true_velocity_model), vmin=0, vmax=3)
#         fig.colorbar(im, ax=ax[1, i],fraction=0.046, pad=0.04)
#         ax[1, i].set_title("Velocity Error {} {}".format(int(50*src[1]), int(50*src[0])))
        
#     plt.tight_layout()
#     plt.savefig("{}/{}.png".format(path, filename))
#     plt.close()
    
    
def IndivPosterior(nsrc, scatter_im, Xsrc, gauss_means, path=None, close=True, bins=201):
""" Plot each histogram as an individual subplot, size == sqrt(nsrc)
    
Arguments: 
    nsrc: number of sources to plot
    scatter_im: all sample points
    Xsrc: source locations matrix
    gauss_means: mean of prior
    path: output folder
    close: close figure if true
    bins: num bins for histogram

"""
    size= int(np.sqrt(nsrc))

    cmap_list=get_cmap(nsrc)
    cum_hist = np.zeros([bins, bins])
    all_hist = np.zeros([nsrc, bins, bins])

    fig, ax = plt.subplots(size, int(np.ceil(nsrc/size)), figsize=(2*size, 2*int(np.ceil(nsrc/size))))
    for i in range(np.min([size**2, nsrc])):
        color = np.ndarray.tolist(np.array(cmap_list(i)).reshape(1,-1))
        pts = scatter_im[:, i, :]

        vals_im = np.histogram2d(pts[:, 1], pts[:, 0], range=[[0, 1], [0,1]], bins=[bins, bins])
        all_hist[i, :, :] = vals_im[0]
        cum_hist += vals_im[0]
    #     print(vals_im)
        ax[i//size, i%size].imshow(vals_im[0].transpose())
        ax[i//size, i%size].scatter(Xsrc[i, 1]*bins, Xsrc[i, 0]*bins,  marker="*", c="yellow", label="True Sources")
        if gauss_means[i, 1]<1 and gauss_means[i, 0] < 1 and gauss_means[i, 1]>0 and gauss_means[i, 0] >0:
            ax[i//size, i%size].scatter(gauss_means[i, 1]*bins, 
                                    gauss_means[i, 0]*bins, alpha=0.3, 
                                    c="red") 
        ax[i//size, i%size].axis("off")
    if path is not None:
        plt.savefig("{}/IndivPosteriorHistograms.png".format(path))
    if close == True:
        plt.close()
    return
    
    
def GenerateDensityMap(nsrc, Xsrc, scatter_im, gauss_means, scale=5, path=None, close=True, bins=201, k=None, k_sub = None):
"""Plot Density Map of all Sources
    
Arguments: 
    nsrc: number of sources to plot
    Xsrc: source locations matrix
    scatter_im: all sample points
    gauss_means: mean of prior
    scale: scale values for histogram
    path: output folder
    close: close figure if true
    bins: num bins for histogram
    k: epoch number/ iteration number for EM
    k_sub: sub epoch number

"""
    all_hist = np.ones([nsrc, bins, bins])
    
    color_order = np.random.choice(nsrc, size=nsrc, replace=False)
    img = np.zeros([3, bins, bins])
    cmap = cmap_list=get_cmap(nsrc, name="hsv")
    for i in range(nsrc):
        color = np.ndarray.tolist(np.array(cmap_list(color_order[i])).reshape(1,-1))
        pts = scatter_im[:, i, :]
        vals_im = np.histogram2d(pts[:, 1], pts[:, 0], range=[[0, 1], [0,1]], bins=[bins, bins])
        all_hist[i, :, :] = vals_im[0]
        for j in range(3):
            img[j, :, :] += color[0][j]*(all_hist[i, :, :])
            
    cum_hist = np.sum(all_hist, axis=0)


    plt.figure()
    plt.imshow(img.transpose()*scale, vmin=0, vmax=img.max())
    for i in range(nsrc):
        color = np.ndarray.tolist(np.array(cmap_list(color_order[i])).reshape(1,-1))
        plt.scatter(Xsrc[i, 1]*bins, 
                    Xsrc[i, 0]*bins, s=10, marker="*",
                    c=color) 
    errtot = 0
    for i in range(nsrc):
        pts = scatter_im[:, i, :]
        err = np.mean(np.abs(pts-Xsrc[i, :])**2)
        errtot += err

    plt.xlim([0, bins-1])
    plt.ylim([bins-1, 0])
    plt.title("MSE {}".format(errtot/nsrc))
    plt.axis("off")
    
    if path is not None:
        if k is None:
            plt.savefig("{}/PosteriorHistograms.png".format(path))
        else:
            plt.savefig("{}/PosteriorHistograms{}_{}.png".format(path,str(k).zfill(5),str(k_sub).zfill(5)))
            
    if close == True:
        plt.close()
    return

def PlotScatterAll(nsrc, nrec, scatter_im, Xrec, Xsrc, mean_img, gauss_means, 
                   px_randmean, prior_sigma, alpha=0.02, k=None, k_sub=None, plot_otherpts=True, path=None, close=True):
"""Plot scatter points
    
Arguments: 
    nsrc: number of sources to plot
    nrec: number of receivers to plot
    scatter_im: all sample points
    Xsrc: source locations matrix
    Xrec: receiver locations matrix
    mean_img: mean of sampled values for posterior
    gauss_means: mean of prior
    px_randmean: plot prior if prior is random
    prior_sigma: std of source prior p(x)
    alpha: scatter point weight
    k: epoch number/ iteration number for EM
    k_sub: sub epoch number
    plot_otherpts: plot p(x) mean, p(x) 3*std, recon mean
    path: output folder
    close: close figure if true
   
"""
    cmap_list=get_cmap(nsrc)
    color_order = np.random.choice(nsrc, size=nsrc, replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    if nrec <= 20:
        plt.scatter(Xrec[:, 1], Xrec[:, 0], c="g", label = "Receivers")
    for plt_iter in range(nsrc):
        color = np.ndarray.tolist(np.array(cmap_list(color_order[plt_iter])).reshape(1,-1))
        if plt_iter == 0:
            plt.scatter(scatter_im[:, plt_iter, 1], scatter_im[:, plt_iter, 0], marker="+", 
                        alpha=alpha,c=color, label="Recon Points")
            if plot_otherpts == True:
                plt.scatter(Xsrc[plt_iter, 1], Xsrc[plt_iter, 0], s=50, marker="*", c=color, label="True Sources")
                plt.scatter(mean_img[plt_iter, 1], mean_img[plt_iter, 0], alpha=0.5, marker="v", 
                        c=color, label = "Mean Recon Source")
                if px_randmean == True:
                    size = (330*prior_sigma*3*2)**2
                    if prior_sigma < 5e-2 or nsrc < 20:
                        plt.scatter(gauss_means[plt_iter, 1], 
                                    gauss_means[plt_iter, 0], alpha=0.03, 
                                    c=color,s=size, label = "Prior Source 3 sigma") #0.3 before
                    else:
                        plt.scatter(gauss_means[plt_iter, 1], 
                                    gauss_means[plt_iter, 0], alpha=0.3, 
                                    c=color, label = "Prior Source Mean") 
        else:
            plt.scatter(scatter_im[:, plt_iter, 1], scatter_im[:, plt_iter, 0], marker="+", 
                        alpha=alpha,c=color)
            if plot_otherpts == True:
                plt.scatter(Xsrc[plt_iter, 1], Xsrc[plt_iter, 0], s=50, marker="*", c=color)
                plt.scatter(mean_img[plt_iter, 1], mean_img[plt_iter, 0], alpha=0.5, marker="v", 
                            c=color)
                if px_randmean == True:
                    size = (330*prior_sigma*3*2)**2
                    if prior_sigma < 5e-2 or nsrc < 20:
                        plt.scatter(gauss_means[plt_iter, 1], 
                                    gauss_means[plt_iter, 0], alpha=0.03, 
                                    c=color,s=size) #0.3 before
                    else:
                        plt.scatter(gauss_means[plt_iter, 1], 
                                    gauss_means[plt_iter, 0], alpha=0.3, 
                                    c=color) 
                        
    errtot = 0
    for i in range(nsrc):
        pts = scatter_im[:, i, :]
        err = np.mean(np.abs(pts-Xsrc[i, :])**2)
        errtot += err
        
    plt.legend(loc=[1.3, 0.5])
    plt.xlim([0, 1])
    plt.ylim([1, 0])
    plt.title("E Step MSE {}".format(errtot/nsrc))
    
    if path is not None:
        if k is None:
            plt.savefig("{}/SourceReconScatter.png".format(path))
        else:
            plt.savefig("{}/SourceReconScatter{}_{}.png".format(path,str(k).zfill(5),str(k_sub).zfill(5)))
    if close == True:
        plt.close()
    return 


def generate_mean_velo(Xrec, FNet, device, VTrue, use_dataparallel, s=51, path=None, k=None, close=True):
""" plot mean velocity
    
Arguments: 
    Xrec: receiver locations matrix
    FNet: forward model network
    device: network device
    VTrue: true velocity model
    use_dataparallel: if device is on dataparallel
    s: size of recon velocity
    path:  output folder
    k: epoch number/ iteration number for EM
    close: close figure if true

"""
    nrec = Xrec.shape[0]
    all_imgs = np.zeros([nrec, s, s])
    mse_all = []
    if nrec >= 4:
        fig, ax = plt.subplots(4, nrec//4, figsize=(10, 8))
    
    for j in range(nrec):
        V, _ = VeloRecon(FNet, device, src=Tensor(Xrec[j, :]), num = s, use_dataparallel=use_dataparallel)
        V = V.detach().cpu().numpy().reshape([s,s]).transpose()
        all_imgs[j, :, :] = V
        mse = np.mean(np.abs(V-VTrue)**2)
        if nrec >= 4:
            ax[j%4, j//4].imshow(V, vmin=0, vmax=10)
            ax[j%4, j//4].scatter(Xrec[j, 1]*51, Xrec[j, 0]*51-2,marker="v", c="red")
            ax[j%4, j//4].axis("off")
            ax[j%4, j//4].set_title(r"MSE {:.2f}".format(mse))
        mse_all.append(mse)
    if path is not None:
        if k is None:
            plt.savefig("{}/AllVelo.png".format(path))
        else:
            plt.savefig("{}/AllVelo{}.png".format(path,str(k).zfill(5)))
    if close == True:
        plt.close()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    # generate points
    mean_img = np.mean(all_imgs, axis=0)
    mse = np.mean(mse_all)
    std = np.std(mse_all)
    plt.imshow(mean_img, vmin=0, vmax=10)
    plt.title(r"MSE {:.2f} $\pm$ {:.2f}".format(mse, std))
    plt.axis("off")
    if path is not None:
        if k is None:
            plt.savefig("{}/MeanVelo.png".format(path))
            np.save("{}/Data/MeanVelo.png".format(path), mean_img)
        else:
            plt.savefig("{}/MeanVelo{}.png".format(path,str(k).zfill(5)))
            np.save("{}/Data/MeanVelo{}.png".format(path,str(k).zfill(5)), mean_img)
    if close == True:
        plt.close()
    return
            
            
######################################################################################################################

        
#                            SETUP HELPERS
    

######################################################################################################################

def GetTrueVelo(model):
""" load true velocity model
    
Arguments: 
    model: name of model

"""
    if model == "Fault":
        true_velocity_model = np.load("SeismoData/GridFault_V.npy")
    elif model == "Blur0":
        true_velocity_model = np.load("SeismoData/GridLayersBlurred1D_0_V.npy")
    elif model == "Blur1":
        true_velocity_model = np.load("SeismoData/GridLayersBlurred1D_1_V.npy")
    elif model == "Blur2":
        true_velocity_model = np.load("SeismoData/GridLayersBlurred1D_2_V.npy")
    elif model == "Blur3":
        true_velocity_model = np.load("SeismoData/GridLayersBlurred1D_3_V.npy")
    elif model == "Blur4":
        true_velocity_model = np.load("SeismoData/GridLayersBlurred1D_4_V.npy")
    elif model == "Blur5":
        true_velocity_model = np.load("SeismoData/GridLayersBlurred1D_5_V.npy")
    elif model == "Gradient":
        true_velocity_model = np.load("SeismoData/GridGradient_V.npy")
    elif model == "Blur5Missing":
        true_velocity_model = np.load("SeismoData/GridLayersBlurred1DMissing_5_V.npy")
    elif model == "Blob1":
        true_velocity_model = np.load("SeismoData/GridBlobLayered_0.1_V.npy")
    elif model == "Blob2":
        true_velocity_model = np.load("SeismoData/GridBlobLayered_0.2_V.npy")
    elif model == "Blob3":
        true_velocity_model = np.load("SeismoData/GridBlobLayered_0.3_V.npy")
    elif model == "Blur2Blob1":
        true_velocity_model = np.load("SeismoData/GridBlobBlur2_0.1_V.npy")
    elif model == "Blur2Blob2":
        true_velocity_model = np.load("SeismoData/GridBlobBlur2_0.2_V.npy")
    elif model == "Blur2Blob3":
        true_velocity_model = np.load("SeismoData/GridBlobBlur2_0.3_V.npy")
    elif model == "Blur3Blob1":
        true_velocity_model = np.load("SeismoData/GridBlobBlur3_0.1_V.npy")
    elif model == "Blur3Blob2":
        true_velocity_model = np.load("SeismoData/GridBlobBlur3_0.2_V.npy")
    elif model == "Blur3Blob3":
        true_velocity_model = np.load("SeismoData/GridBlobBlur3_0.3_V.npy")
    elif model == "Blur4Blob1":
        true_velocity_model = np.load("SeismoData/GridBlobBlur4_0.1_V.npy")
    elif model == "Blur4Blob2":
        true_velocity_model = np.load("SeismoData/GridBlobBlur4_0.2_V.npy")
    elif model == "Blur4Blob3":
        true_velocity_model = np.load("SeismoData/GridBlobBlur4_0.3_V.npy")

        
    elif model == "GradBlur1.3":
        true_velocity_model = np.load("SeismoData/GradBlur1.3_V.npy")
    elif model == "GradBlur1.3Grad":
        true_velocity_model = np.load("SeismoData/GradBlur1.3Grad_V.npy")
        print("true velo is grid")
    elif model == "GradBlur1.3Blob3":
        true_velocity_model = np.load("SeismoData/GradBlobBlur1.3_0.3_V.npy")

    elif model == "GradBlobBlur":
        true_velocity_model = np.load("SeismoData/GradBlobBlur_V.npy")
    elif model == "GradBlobBlur0":
        true_velocity_model = np.load("SeismoData/GradBlobBlur0_V.npy")
    elif model == "GradBlobBlur3":
        true_velocity_model = np.load("SeismoData/GradBlobBlur3_V.npy")

    elif model == "NewGrad":
        true_velocity_model = np.load("SeismoData/NewGrad_V.npy")
    elif model == "NewLayers0":
        true_velocity_model = np.load("SeismoData/NewLayers0_V.npy")
    elif model == "NewLayersBlob0":
        true_velocity_model = np.load("SeismoData/NewLayersBlob0_V.npy")
    elif model == "NewLayersBlob3":
        true_velocity_model = np.load("SeismoData/NewLayersBlob3_V.npy")

    elif model == "VGRF0_v2":
        true_velocity_model = np.load("SeismoData/VGRF0_v2_V.npy")
    elif model == "VGRF1_v2":
        true_velocity_model = np.load("SeismoData/VGRF1_v2_V.npy")
    elif model == "VGRF2_v2":
        true_velocity_model = np.load("SeismoData/VGRF2_v2_V.npy")
    elif model == "VGRF3_v2":
        true_velocity_model = np.load("SeismoData/VGRF3_v2_V.npy")
    elif model == "VGRF4_v2":
        true_velocity_model = np.load("SeismoData/VGRF4_v2_V.npy")
    elif model == "VGRF5_v2":
        true_velocity_model = np.load("SeismoData/VGRF5_v2_V.npy")
    elif model == "VGRF6_v2":
        true_velocity_model = np.load("SeismoData/VGRF6_v2_V.npy")
    elif model == "VGRF7_v2":
        true_velocity_model = np.load("SeismoData/VGRF7_v2_V.npy")
    elif model == "VGRF8_v2":
        true_velocity_model = np.load("SeismoData/VGRF8_v2_V.npy")
    elif model == "VGRF9_v2":
        true_velocity_model = np.load("SeismoData/VGRF9_v2_V.npy")

    elif model == "H5":
        true_velocity_model = np.ones([samples, samples])*5
    elif model == "H6":
        true_velocity_model = np.ones([samples, samples])*6
    elif model == "H5.5":
        true_velocity_model = np.ones([samples, samples])*5.5
    else:
        true_velocity_model = np.load("SeismoData/GridLayered_V.npy")
    return true_velocity_model


######################################################################################################################

        
#                            Plotting
    

######################################################################################################################

def generate_tt(Xrec_idx, Xsrc_idx, V, x):
"""  generate travel time field of size nsrc x nrec
    
Arguments: 
    Xrec_idx: index of receivers in V
    Xsrc_idx: index of sources in V
    x: linspace for V

"""
    nsrc = Xsrc_idx.shape[0]
    nrec = Xrec_idx.shape[0]
    tt = np.zeros([nsrc, nrec])
    spacing = x[1]-x[0]
    
    X, Y = np.meshgrid(x, x)
    dx = (spacing, spacing)
    
    for i in range(nsrc):
        src      = Xsrc_idx[i, :] 
        order = 2

        tau_ffm = eikonalfm.factored_fast_marching(V, src, dx, order)
        
        T0 = eikonalfm.distance(tau_ffm.shape, dx, src, indexing="ij")
        TT = (tau_ffm*T0)
        
        tt[i, :] = TT[Xrec_idx[:, 0], Xrec_idx[:, 1]]
    return tt, TT

def generate_tt_homogeneous(Xrec, Xsrc, v, use_torch=False):
""" generate travel time field for homogeneous velocity
    
Arguments: 
    Xrec: receiver locations
    Xsrc: source locations
    v: velocity value
    use_torch: use pytorch vs numpy

"""
    if use_torch==False:
        nsrc = Xsrc.shape[0]
        nrec = Xrec.shape[0]
        Xsrc = np.repeat(np.expand_dims(Xsrc, axis=1), nrec, axis=1)
        Xrec = np.repeat(np.expand_dims(Xrec, axis=0), nsrc, axis=0)
        D = np.sqrt(np.sum((Xrec - Xsrc)**2, axis=2))
    else:
        # input: [nbatch, nsrc/nrec, d]
        # output: [nbatch, nsrc, nrec, d]
        nsrc = Xsrc.shape[1]
        nrec = Xrec.shape[1]
        Xrec = torch.cat(nsrc*[torch.unsqueeze(Xrec, axis=1)], dim=1)
        Xsrc = torch.cat(nrec*[torch.unsqueeze(Xsrc, axis=2)], dim=2)
        D = torch.sqrt(torch.sum((Xrec - Xsrc)**2, dim=3))
    return D/v

def VeloRecon(network, device, src=[0.5], num = 20, use_dataparallel=False):
""" generate velocity reconstruction of size num x num with respect to source src
    
Arguments: 
    network: eikonet
    device: network device
    src: source location
    num: size of velocity
    use_dataparallel: network on dataparallel

"""
    x = np.linspace(0, 1, num)
    X, Y = np.meshgrid(x, x)
    if len(src) > 1:
        Xsrc = Tensor(src[np.newaxis, :]).to(device)
    else:
        Xsrc = torch.ones([1, 2], device = device)*src[0]
    Xrec = np.zeros([X.size, 2])
    Xrec[:, 0] = X.flatten(); Xrec[:, 1] = Y.flatten()
    Xrec = Tensor(Xrec).requires_grad_().to(device)
    if use_dataparallel == True:
        tau, _, _, _, _ = network.module.generate_tau(torch.unsqueeze(Xsrc, axis=0), torch.unsqueeze(Xrec, axis=0))
    else:
        tau, _, _, _, _ = network.generate_tau(torch.unsqueeze(Xsrc, axis=0), torch.unsqueeze(Xrec, axis=0))
    
    dtau  = torch.squeeze(torch.autograd.grad(outputs=tau, inputs=Xrec, grad_outputs=torch.ones(tau.size()).to(device), 
                    only_inputs=True,create_graph=True,retain_graph=False)[0], axis=0)
    tau = torch.squeeze(tau,axis=1)

    T0    = torch.sqrt(torch.sum((Xsrc - Xrec)**2, axis=1)+1e-8)  
    T1    = (T0**2)*(torch.sum(dtau**2, axis=1))
    T2    = 2*tau*(dtau[:,0]*(Xrec[:,0]-Xsrc[:,0]) + dtau[:,1]*(Xrec[:,1]-Xsrc[:,1]))
    T3    = tau**2
    S2    = (T1+T2+T3)
    V = torch.sqrt(1/S2+1e-8)
    return V, tau*T0

def VeloReconVNet(VNet, device, num = 20, vmat=True):
""" generate velocity reconstruction of size num x num with respect to source src
    
Arguments: 
    VNet: network for velocity model
    device: network device
    num: size of velocity
    vmat: velocity network as a learned matrix

"""
    if vmat == True:
        return VNet.V.transpose(0, 1)
    else:
        x = np.linspace(0, 1, num)
        X, Y = np.meshgrid(x, x)

        Xrec = np.zeros([1, X.size, 2])
        Xrec[:, :, 0] = X.flatten(); Xrec[:, :, 1] = Y.flatten()
        Xrec = Tensor(Xrec).requires_grad_().to(device)

        V = VNet(Xrec)
        V = V.reshape([num, num])
        return V


def generate_gif(DIR, name, epoch, subepoch, s, subs):
""" generate gif of a certain filename
    
Arguments: 
    DIR: output directory
    name: filename
    epoch: epoch (or EM iteration)
    subepoch: E/M step epoch
    s: epoch spacing
    subs: subepoch spacing

"""
    images = []
    for i in range(0, epoch, s):
        for j in range(0, subepoch, subs):
            filename = '{}/{}{}_{}.png'.format(DIR, name, str(i).zfill(5),str(j).zfill(5))
            images.append(imageio.imread(filename))
    imageio.mimsave('{}/{}.gif'.format(DIR, name), images)
    return