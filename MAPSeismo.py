#!/usr/bin/env python
# coding: utf-8

# import psutil
# psutil.Process().nice(5)

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
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler,WeightedRandomSampler
print(torch.__version__)

from scipy.signal import convolve2d
print(scipy.__version__)
from generative_model import realnvpfc_model
from seismo_helpers import *
from seismo_models import *
import argparse
import matplotlib 

matplotlib.rcParams['figure.dpi'] = 200

torch.autograd.set_detect_anomaly(True)

    
def main_function(args):

    d = 2
#     tv=1e3
    n_flow = 16
    samples = 51
    logdet = 1
    affine = True
    data_weight = args.data_weight/args.data_sigma**2
#     data_weight = args.data_weight/args.data_sigma**2
#     phi_weight = args.phi_weight
    use_bias = False
#     prior_weight = args.prior_weight

    ################################################ SET UP DATA ####################################################
    if args.load == True:
        args.nsrc = 4
        args.nrec = 5
        d = 2
        true_velocity = 4
        
        Xsrc = np.load("SeismoData/GridXsrc.npy")
        Xrec = np.load("SeismoData/GridXrec.npy")
        Xsrc_idx = np.load("SeismoData/GridXsrc_idx.npy")
        Xrec_idx = np.load("SeismoData/GridXrec_idx.npy")

        fwd_velocity_model = np.load("SeismoData/GridFwd_V.npy")
        fwd_tt = np.load("SeismoData/GridFwd_TT.npy")

        true_velocity_model = np.load("SeismoData/GridTrue_V.npy")
        true_tt = np.load("SeismoData/GridTrue_tt.npy")
    else:
        x = np.linspace(0, 1, samples)
        idx = np.linspace(0, samples-1, samples)[1:-1]

        Xrec_idx = np.random.choice(idx, [args.nrec, d]).astype(np.int32)
        if args.surfaceR != 0:
            gridpts = np.linspace(0, samples-1, args.nrec//args.surfaceR+2)[1:-1]
        if args.surfaceR == 1: # constrain receivers to surface 
            Xrec_idx[:, 1] = gridpts
            Xrec_idx[:, 0] = 0
        elif args.surfaceR == 2:
            Xrec_idx[0:args.nrec//2, 1] = gridpts
            Xrec_idx[0:args.nrec//2, 0] = 0
            Xrec_idx[args.nrec//2:, 0] = gridpts
            Xrec_idx[args.nrec//2:, 1] = 0
        elif args.surfaceR == 3:
            Xrec_idx[0:args.nrec//3, 1] = gridpts
            Xrec_idx[0:args.nrec//3, 0] = 0
            Xrec_idx[args.nrec//3:2*args.nrec//3, 0] = gridpts
            Xrec_idx[args.nrec//3:2*args.nrec//3, 1] = 0
            Xrec_idx[2*args.nrec//3:, 1] = gridpts
            Xrec_idx[2*args.nrec//3:, 0] = -1
        elif args.surfaceR == 4:
            Xrec_idx[0:args.nrec//4, 1] = gridpts
            Xrec_idx[0:args.nrec//4, 0] = 0
            Xrec_idx[args.nrec//4:2*args.nrec//4, 0] = gridpts
            Xrec_idx[args.nrec//4:2*args.nrec//4, 1] = 0
            Xrec_idx[2*args.nrec//4:3*args.nrec//4, 1] = gridpts
            Xrec_idx[2*args.nrec//4:3*args.nrec//4, 0] = -1
            Xrec_idx[3*args.nrec//4:, 0] = gridpts
            Xrec_idx[3*args.nrec//4:, 1] = -1
        Xrec = x[Xrec_idx]

        plotRecIdx = np.random.choice(args.nrec,  4)
        plotRec = Xrec[plotRecIdx, :]
       
        if args.nsrc == 1 and args.center == True:
            Xsrc_idx = (np.random.choice(idx, [args.nsrc, d])*0 + samples//2).astype(np.int32)
            Xsrc = x[Xsrc_idx]
        else:
            if args.gridsrcs == True:
                xsamples = int(np.round(args.nsrc**(1/2)))
                xidx = np.linspace(0, samples-1, xsamples)
                yidx = np.linspace(0, samples-1, xsamples+1)[1:]
                X, Y = np.meshgrid(yidx, xidx)
                Xsrc_idx = np.zeros([len(xidx)**2, 2])
                Xsrc_idx[:, 0] = X.flatten(); Xsrc_idx[:, 1] = Y.flatten()
                Xsrc_idx = np.round(Xsrc_idx).astype(np.int32)
                Xsrc = x[Xsrc_idx]
                args.nsrc = Xsrc.shape[0]
            else:
                Xsrc_idx = np.random.choice(idx, [args.nsrc, d]).astype(np.int32)
                Xsrc = x[Xsrc_idx]

        np.save("{}/Data/GridXsrc.npy".format(args.PATH), Xsrc)
        np.save("{}/Data/GridXrec.npy".format(args.PATH), Xrec)
        np.save("{}/Data/GridXsrc_idx.npy".format(args.PATH), Xsrc_idx)
        np.save("{}/Data/GridXrec_idx.npy".format(args.PATH), Xrec_idx)

        
        # LOAD TRUE VELOCITY MODEL AS 51 x 51 matrix
        true_velocity_model = GetTrueVelo(args.model)
        
        if args.fwdmodel is None:
            fwd_velocity = np.mean(true_velocity_model)
            print("Fwd Velocity is {}", fwd_velocity)
            fwd_velocity_model = np.ones([samples, samples])*fwd_velocity
        else:
            fwd_velocity_model = np.load("SeismoData/{}_V.npy".format(args.fwdmodel))
            fwd_velocity= 5

        np.save("{}/Data/GridFwd_VModel.npy".format(args.PATH), fwd_velocity_model)
        np.save("{}/Data/GridTrue_VModel.npy".format(args.PATH), true_velocity_model)
        
        fwd_tt, _ = generate_tt(Xrec_idx, Xsrc_idx, fwd_velocity_model, x)
        np.save("{}/Data/GridFwd_TT.npy".format(args.PATH), fwd_tt)

        true_tt_nonoise, TT = generate_tt(Xrec_idx, Xsrc_idx, true_velocity_model, x)
        np.save("{}/Data/GridTrue_TT.npy".format(args.PATH), true_tt_nonoise)

        if args.nonoise == False:
            tt_addednoise_err = np.random.normal(0, args.data_sigma, true_tt_nonoise.shape)
        else:
            tt_addednoise_err = 0
        true_tt = true_tt_nonoise +  tt_addednoise_err 
        np.save("{}/Data/GridTrueNoiseAdded_TT.npy".format(args.PATH), true_tt)
        np.save("{}/Data/GridTT.npy".format(args.PATH), TT)
        
        output_matrix = np.zeros([args.nsrc*args.nrec, 5])
        for i in range(args.nsrc):
            for j in range(args.nrec):
                output_matrix[i*args.nrec+j, 0:2] = Xsrc[i, :]
                output_matrix[i*args.nrec+j, 2:4] = Xrec[j, :]
                output_matrix[i*args.nrec+j, 4] = true_tt[i, j]
        np.save("{}/Data/SrcRecTT.npy".format(args.PATH), output_matrix)
        np.savetxt("{}/Data/SrcRecTT.txt".format(args.PATH), output_matrix, delimiter="\t")
        
        gauss_means = Xsrc + np.random.normal(0, args.prior_sigma, Xsrc.shape)
        np.save("{}/Data/GridXsrcPrior.npy".format(args.PATH), gauss_means)
        
        output_matrix = np.zeros([args.nsrc*args.nrec, 5])
        for i in range(args.nsrc):
            for j in range(args.nrec):
                output_matrix[i*args.nrec+j, 0:2] = gauss_means[i, :]
                output_matrix[i*args.nrec+j, 2:4] = Xrec[j, :]
                output_matrix[i*args.nrec+j, 4] = true_tt[i, j]
        np.save("{}/Data/PriorSrcRecTT.npy".format(args.PATH), output_matrix)
        np.savetxt("{}/Data/PriorSrcRecTT.txt".format(args.PATH), output_matrix, delimiter="\t")
        
        
        if args.velo_loss == True:
            print("Generating Velocity Truth Signal")
            fwd_tt = np.concatenate(args.nsrc*[fwd_velocity_model[Xrec_idx[:, 0], Xrec_idx[:, 1]][np.newaxis, :]], axis=0)
            true_tt = np.concatenate(args.nsrc*[true_velocity_model[Xrec_idx[:, 0], Xrec_idx[:, 1]][np.newaxis, :]], axis=0)
            np.save("{}/Data/GridFwd_VTrain.npy".format(args.PATH), fwd_tt)
            np.save("{}/Data/GridTrue_VTrain.npy".format(args.PATH), true_tt)

    ####################################### SET UP FIGURES ###################################################

    fig, ax = plt.subplots(1, 4, figsize = (12, 3))
    ax[1].scatter(Xrec[:, 1], Xrec[:, 0])
    ax[1].scatter(Xsrc[:, 1], Xsrc[:, 0])
    
    ax[1].axis('square')
    ax[1].set_xlim([0, 1])
    ax[1].set_ylim([1, 0])
    ax[1].legend(["Reciever", "Source"])
    
    im=ax[0].matshow(true_tt.transpose())
    ax[0].set_ylabel("receivers")
    ax[0].set_xlabel("sources")
    fig.colorbar(im, ax=ax[1],fraction=0.046, pad=0.04)
    im = ax[2].matshow(TT, vmin = 0, vmax=0.2)
    ax[2].scatter(Xsrc_idx[-1, 1], Xsrc_idx[-1, 0])
    ax[2].scatter(Xrec_idx[:, 1], Xrec_idx[:, 0])
    fig.colorbar(im, ax=ax[2],fraction=0.046, pad=0.04)
    ax[3].matshow(true_velocity_model, vmin = 0, vmax = 10)

    ax[1].set_title("Sources and Receivers Configuration")
    ax[0].set_title("Travel Time Grid")
    ax[2].set_title("Travel Time Field of Example")
    ax[3].set_title("Velocity Field")
    plt.tight_layout()
    
    plt.savefig("{}/Setup.png".format(args.PATH))
    plt.close()
    
    
    fig, ax = plt.subplots(2, 4, figsize = (13, 5))
    im=ax[0,0].imshow(true_velocity_model, vmin=0, vmax=10)
    fig.colorbar(im, ax=ax[0,0],fraction=0.046, pad=0.04)
    im=ax[0,1].imshow(fwd_velocity_model, vmin=0, vmax=10)
    fig.colorbar(im, ax=ax[0,1],fraction=0.046, pad=0.04)
    im=ax[0,2].imshow(np.abs(true_velocity_model-fwd_velocity_model), vmin=0, vmax=2)
    fig.colorbar(im, ax=ax[0,2],fraction=0.046, pad=0.04)

    im=ax[1,0].imshow(true_tt.transpose())
    fig.colorbar(im, ax=ax[1,0],fraction=0.046, pad=0.04)
    im=ax[1,1].imshow(fwd_tt.transpose())
    fig.colorbar(im, ax=ax[1,1],fraction=0.046, pad=0.04)
    im=ax[1,2].imshow(np.abs(true_tt - fwd_tt).transpose())
    fig.colorbar(im, ax=ax[1,2],fraction=0.046, pad=0.04)
    im=ax[1,3].imshow(np.abs(true_tt - fwd_tt).transpose()/(true_tt+1e-9).transpose())
    fig.colorbar(im, ax=ax[1,3],fraction=0.046, pad=0.04)

    for i in range(4):
        ax[1, i].set_xlabel("Source")
        ax[1, i].set_ylabel("Receiver")

    ax[0,3].axis("off")

    ax[0,0].set_title("True Velocity")
    ax[0,1].set_title("Fwd Velocity")
    ax[0,2].set_title("Absolute Difference")

    ax[1,0].set_title("True Travel Time")
    ax[1,1].set_title("Fwd Travel Time")
    ax[1,2].set_title("Absolute Difference")
    ax[1,3].set_title("Relative Difference")

    plt.tight_layout()
    plt.savefig("{}/VelocitySetup.png".format(args.PATH))
    plt.close()
    
    
    ############################################## MODEL SETUP #####################################################

    print(Xsrc.shape)
    Xsrc = Tensor(Xsrc).to(args.device)
    Xrec = Tensor(Xrec).to(args.device)
    true_tt = Tensor(true_tt).to(args.device)
    true_tt_nonoise = Tensor(true_tt_nonoise).to(args.device)
    fwd_tt = Tensor(fwd_tt).to(args.device)
    gauss_means = Tensor(gauss_means).to(args.device)
        
    if args.nsrc < 4:
        seqfrac = 1
    else:
        seqfrac = 4
    GNet = realnvpfc_model.RealNVP(d*args.nsrc, n_flow, affine=affine, seqfrac = seqfrac).to(args.device)


    if args.use_invareiko:
        print("Using EikoNetInvar")
        FNet = EikoNetInvar(input_size = d, sine_activation=args.sine_activation, sine_freq = args.sine_freqs)
        FNet.init_model(args.fwdmodel, args.randinit, args.device)
    elif args.use_eikonet == True:
        if args.ffreqs is None:
            if args.large == True:
                print("Using Eikonet8()")
                FNet = EikoNet8(input_size = d, sine_activation=args.sine_activation, sine_freq = args.sine_freqs)
                FNet.init_model(args.fwdmodel, args.randinit, args.device)
            else:
                print("Using Eikonet()")
                FNet = EikoNet(input_size = d, sine_activation=args.sine_activation, sine_freq = args.sine_freqs)
                FNet.init_model(args.fwdmodel, args.randinit, args.device, init_prior=False)
        else:
            print("Using Fourier Freqs")
            FNet = FreqEikoNet(input_size = d, num_freqs = args.ffreqs, sine_activation=args.sine_activation,
                                    sine_freq = args.sine_freqs)
            FNet.apply(init_weights_eiko_ffreqs)
    else:
        print("Using MLP()")
        FNet = MLPSrcRec(input_size = d)
        FNet.apply(init_weights)
        

    if args.use_dataparallel == True:
        print("Parallel Training with {} GPUS".format(len(args.device_ids)))
        GNet = nn.DataParallel(GNet, device_ids = args.device_ids)
        GNet.to(args.device)
        FNet = nn.DataParallel(FNet, device_ids = args.device_ids)
        FNet.to(args.device) 
       
        
    FNet.to(args.device)

    # Load Forward Model Method
    if args.fwdmodel is None:
        FTrue = lambda Xrec, Xsrc, v: generate_tt_homogeneous(Xrec, Xsrc, v, use_torch=True)
    else:
        output_matrix = np.load("SeismoData/{}_XTTPairs.npy".format(args.fwdmodel))
        FTrue = lambda idx: Tensor(output_matrix[idx, -1][np.newaxis, :, np.newaxis]).to(args.device)
        
    s = 51
    fig, ax = plt.subplots(2, 4, figsize=(14, 8))
    for i in range(0, 4):
        if args.pltRec == True:
            src = plotRec[i, :]
        else:
            src = np.array([0.25*i, 0.25*i])
        V, TT = VeloRecon(FNet, args.device, num = s, src=src, use_dataparallel=args.use_dataparallel)
        V = V.detach().cpu().numpy().reshape([s, s])
        TT = TT.detach().cpu().numpy().reshape([s, s])
        np.save("{}/Data/VInit{}.npy".format(args.PATH, i), V)
        error_im = np.abs(V.transpose()-true_velocity_model)
        err = np.sum(error_im)/51/51
        
        im=ax[0, i].imshow(V.transpose(), vmin=0, vmax=10)
        fig.colorbar(im, ax=ax[0, i],fraction=0.046, pad=0.04)
        ax[0, i].set_title("Velocity {} {} {:.3f}".format(int(50*src[1]), int(50*src[0]), err))
        
        im=ax[1, i].imshow(error_im, vmin=0, vmax=3)
        fig.colorbar(im, ax=ax[1, i],fraction=0.046, pad=0.04)
        ax[1, i].set_title("Velocity Error {} {} ".format(int(50*src[1]), int(50*src[0])))
        
    plt.tight_layout()
    plt.savefig("{}/FwdModelInit.png".format(args.PATH))
    plt.close()

    flux = np.sum(Xsrc.cpu().numpy())

    criterion=nn.MSELoss(reduction=args.reduction)

    logscale_factor = Img_logscale(scale=flux/(0.8*d*args.nsrc)).to(args.device)
    
    Xsrc_ext = torch.cat(args.btsize*[torch.unsqueeze(Xsrc, axis=0)], axis=0)
    n_sample = args.btsize
    Xsrc_ext1024 = torch.cat(n_sample*[torch.unsqueeze(Xsrc, axis=0)], axis=0)
    X0 = Xsrc.detach().cpu().numpy()

    if args.px_randmean == False:
        print("Using True Means for Source Location Prior")
        prior_gauss = lambda x: criterion(x, Xsrc_ext)/args.prior_sigma**2
    else:
        print("Using Random Means for Source Location Prior")
#         gauss_means = torch.normal(Xsrc, args.prior_sigma)
        gauss_means_ext = torch.cat(args.btsize*[torch.unsqueeze(gauss_means, axis=0)], axis=0)
#         np.save("{}/Data/GridXsrcPrior.npy".format(args.PATH), gauss_means.detach().cpu().numpy())
        prior_gauss = lambda x: criterion(x, gauss_means_ext)/args.prior_sigma**2
    prior_unif = lambda x: torch.sum((x < 1)*(x > 0))

    prior = prior_gauss
    print("Gaussian Prior on Xsrc", prior_gauss(Xsrc_ext))

    ### DEFINE OPTIMIZERS 
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,list(GNet.parameters())
                                         +list(logscale_factor.parameters())+list(FNet.parameters())),lr = args.Elr)

    #################################### TRAINING #########################################################


    loss_list = []
    loss_mse_list = []
    loss_phi_list = []
    loss_invar_list = []
    loss_ttinvar_list = []
    loss_prior_list = []
    
    velo_err_list = []
    tt_err_list = []
    tt_true_err_list = []
    source_err_list = []

    z_sample1 = torch.randn(2*args.nsrc).to(device=args.device)
    z_sample = torch.cat(args.btsize*[torch.unsqueeze(z_sample1, axis=0)], axis=0)
    np.save(f"{args.PATH}/Data/zsample.npy", z_sample.detach().cpu().numpy())
    x_sample = torch.randn(1, 2*args.nsrc).to(device=args.device)
    XrecIn = torch.cat(args.btsize*[torch.unsqueeze(Xrec, axis=0)])
    
    if args.velo_loss == True:
        XrecIn.requires_grad = True
    
    true_tt_ext = torch.cat(args.btsize*[torch.unsqueeze(true_tt, axis=0)], axis=0)
    true_tt_nonoise_ext = torch.cat(args.btsize*[torch.unsqueeze(true_tt_nonoise, axis=0)], axis=0)
    
    x, logdet = GForward(z_sample, GNet, nsrc = args.nsrc, d=d, logscale_factor=logscale_factor, 
                         eiko=args.EIKO, xtrue=Xsrc_ext, device=args.device,
                         use_dataparallel=args.use_dataparallel)
    test = FNet(Xsrc=x, Xrec=XrecIn, v = fwd_velocity, device=args.device) 
    print("Check Initialization", torch.min(test), torch.max(test), torch.min(true_tt), torch.max(true_tt))
    
    sample_noise = args.prior_sigma*torch.randn(Xsrc_ext.shape).to(args.device)
    if args.sampled == True:
        np.save("{}/Data/XsrcWithNoise.npy".format(args.PATH), (sample_noise + Xsrc_ext).detach().cpu().numpy())
   
    for k in range(args.num_epochs):      
        for k_sub in range(args.num_subepochsM):

        
            if args.fwdmodel is None:
                x_idx = None
                x_sample_src = torch.randn(args.btsize, args.nsrc, 2).to(device=args.device)
                x_sample_rec = torch.randn(args.btsize, args.nrec, 2).to(device=args.device)
            else:
                x_idx = np.random.choice(output_matrix.shape[0], args.btsize)
                x_sample_src = Tensor(output_matrix[x_idx, 0:2][np.newaxis, :, :]).to(args.device)
                x_sample_rec = Tensor(output_matrix[x_idx, 2:4][np.newaxis, :, :]).to(args.device)
                
                # fwd_velocity_model
                
            # Sources for invariances
            if args.invar_rec == True: # use receivers for sources in the invariance losses
                invar_src = torch.rand(1, 10, 2).to(device=args.device)
                invar_rec_idx = np.random.choice(args.nrec, 10)
                invar_src = Xrec[invar_rec_idx, :][np.newaxis, :]           
            else:
                invar_src = torch.rand(1, 10, 2).to(device=args.device)
            invar_rec = torch.rand(1, 100, 2).to(device=args.device)
            invar_rec.requires_grad = True


            loss, philoss, mse, priorloss, invar, ttinvarloss = MAPStep(z_sample, XrecIn, x_sample_src, x_sample_rec, 
                                                                      args.device, true_tt_ext, GNet, FNet,
                                                                      args.phi_weight, FTrue, args.data_sigma, velocity=fwd_velocity, 
                                                                      fwd_velocity=fwd_velocity, nsrc = args.nsrc, d=d, 
                                                                      logscale_factor=logscale_factor, eiko=False, 
                                                                      xtrue = Xsrc_ext, fwdmodel = args.fwdmodel, 
                                                                      xidx = x_idx, velo_loss=args.velo_loss, 
                                                                      invar_weight = args.invar_weight, 
                                                                      invar_src = invar_src, invar_rec = invar_rec,
                                                                      vnet_weight = args.vnet_weight, ttinvar = args.ttinvar,
                                                                      use_dataparallel = args.use_dataparallel, 
                                                                      fwd_velocity_model = fwd_velocity_model, 
                                                                      sampled=False, prior_x = args.prior_sigma,
                                                                      samplenoise = None,reduction=args.reduction, prior=prior,
                                                                        prior_weight=args.prior_weight,
                                                                        data_weight=data_weight)


            loss_list.append(loss.detach().cpu().numpy())
            loss_mse_list.append(mse)
            loss_phi_list.append(philoss)
            loss_invar_list.append(invar)
            loss_ttinvar_list.append(ttinvarloss)
            loss_prior_list.append(priorloss)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(FNet.parameters())+list(GNet.parameters()), 1)
            optimizer.step()

            V, TT = VeloRecon(FNet, args.device, num = s, src=src, use_dataparallel=args.use_dataparallel)
            V = V.detach().cpu().numpy().reshape([s, s])
            error_im = np.abs(V.transpose()-true_velocity_model)
            err = np.sum(np.abs(error_im)**2)/51/51
            tt_err_list.append(mse)
            velo_err_list.append(err)
            tt_true_err_list.append(np.mean(tt_addednoise_err))
            del V, TT, error_im, err
            
            ## SCATTER PLOT RECONSTRUCTED SOURCES             
            img, logdet = GForward(z_sample, GNet, args.nsrc, d, logscale_factor, 
                                   eiko=args.EIKO, xtrue = Xsrc_ext1024, device=args.device,
                                   use_dataparallel=args.use_dataparallel)
            source_err = nn.MSELoss()(img, Xsrc_ext1024)
            source_err_list.append(source_err.item())
            del img, logdet, source_err
             
            
            # CHECKPOINT
            if ((k_sub%args.save_every == 0) and args.EMFull) or ((k%args.save_every == 0) and  not args.EMFull):
                with torch.no_grad():
                    if args.use_dataparallel == True:
                        torch.save({
                            'epoch':k,
                            'model_state_dict': FNet.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, '{}/{}{}_{}.pt'.format(args.PATH,"ForwardNetwork",str(k).zfill(5),str(k_sub).zfill(5)))
                        torch.save({
                            'epoch':k,
                            'model_state_dict': GNet.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, '{}/{}{}_{}.pt'.format(args.PATH,"GeneratorNetwork",str(k).zfill(5),str(k_sub).zfill(5)))
                    else:
                        torch.save({
                            'epoch':k,
                            'model_state_dict': FNet.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, '{}/{}{}_{}.pt'.format(args.PATH,"ForwardNetwork",str(k).zfill(5),str(k_sub).zfill(5)))
                        torch.save({
                            'epoch':k,
                            'model_state_dict': GNet.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, '{}/{}{}_{}.pt'.format(args.PATH,"GeneratorNetwork",str(k).zfill(5),str(k_sub).zfill(5)))

                # PLOT LOSS
                fig, ax = plt.subplots()
                ax.plot(np.log10(tt_err_list))
                ax.plot(np.log10(velo_err_list))
                ax.plot(np.log10(tt_true_err_list))
                ax.legend(["TT MSE", "Velo MSE", "TT True Error"])
                plt.savefig("{}/TTV_error.png".format(args.PATH))
                plt.close()    
                
#                 PLOT LOSSES
                fig, ax = plt.subplots()
                ax.plot(np.log10(tt_err_list))
                ax.plot(np.log10(tt_true_err_list))
                ax.plot(np.log10(source_err_list))

                ax.legend(["TT MSE", "TT True Error", "Source Error"])
                plt.savefig("{}/SV_error.png".format(args.PATH))
                plt.close()
                
                    
                s = 51
                V, TT = VeloRecon(FNet, args.device, num = s, use_dataparallel=args.use_dataparallel)
                V = V.detach().cpu().numpy().reshape([s, s])
                TT = TT.detach().cpu().numpy().reshape([s, s])
                np.save("{}/Data/VRecon.npy".format(args.PATH), V)
                
#                 ## PLOT VELOCITY RECON
#                 if args.vnet_invar == True or args.vmat_invar == True:
#                     V_vnet = VeloReconVNet(VNet, args.device, num = s, vmat = args.vmat_invar).detach().cpu().numpy()
#                     np.save("{}/Data/VNetRecon.npy".format(args.PATH), V_vnet)
                    
#                     error_im = np.abs(true_velocity_model-V_vnet.transpose())
#                     err = np.sum(np.abs(error_im))/51/51
                    
#                     fig, ax = plt.subplots(1, 7, figsize = (16, 5))
#                     im=ax[5].imshow(V_vnet.transpose(), vmin=0, vmax=10)
#                     fig.colorbar(im, ax=ax[5],fraction=0.046, pad=0.04)
#                     ax[5].set_title("V Network Recon")
                    
#                     im=ax[6].imshow(error_im, vmin=0, vmax=3)
#                     fig.colorbar(im, ax=ax[6],fraction=0.046, pad=0.04)
#                     ax[6].set_title("V Network Recon Error {:.3f}".format(err))
#                 else:
                fig, ax = plt.subplots(1, 5, figsize = (14, 5))
                    
                error_im = np.abs(true_velocity_model-V.transpose())
                err = np.sum(np.abs(error_im))/51/51

                im=ax[0].imshow(true_velocity_model, vmin=0, vmax=10)
                fig.colorbar(im, ax=ax[0],fraction=0.046, pad=0.04)
                im=ax[1].imshow(fwd_velocity_model, vmin=0, vmax=10)
                fig.colorbar(im, ax=ax[1],fraction=0.046, pad=0.04)
                im=ax[2].imshow(V.transpose(), vmin=0, vmax=10)
                fig.colorbar(im, ax=ax[2],fraction=0.046, pad=0.04)
                im=ax[3].imshow(error_im, vmin=0, vmax=3)
                fig.colorbar(im, ax=ax[3],fraction=0.046, pad=0.04)
                im=ax[4].imshow(TT.transpose(), vmin=0, vmax=0.3)
                fig.colorbar(im, ax=ax[4],fraction=0.046, pad=0.04)
                
                ax[0].set_title("True Velocity")
                ax[1].set_title("Fwd Velocity")
                ax[2].set_title("Velocity Recon")
                ax[3].set_title("Recon Error {:.3f}".format(err))
                ax[4].set_title("Travel Time Field")
                plt.tight_layout()
                plt.savefig("{}/VelocityRecon{}_{}.png".format(args.PATH,str(k).zfill(5),str(k_sub).zfill(5)))
                plt.close()
                
                fig, ax = plt.subplots(2, 4, figsize=(14, 8))
                for i in range(0, 4):
                    if args.pltRec == True:
                        src = plotRec[i, :]
                    else:
                        src = np.array([0.25*i, 0.25*i])
                    V, TT = VeloRecon(FNet, args.device, num = s, src=src, use_dataparallel=args.use_dataparallel)
                    V = V.detach().cpu().numpy().reshape([s, s])
                    TT = TT.detach().cpu().numpy().reshape([s, s])
                    error_im = np.abs(V.transpose()-true_velocity_model)
                    err = np.sum(np.abs(error_im))/51/51
                    np.save("{}/Data/VFinalRecon{}.npy".format(args.PATH, i), V)

                    im=ax[0, i].imshow(V.transpose(), vmin=0, vmax=10)
                    fig.colorbar(im, ax=ax[0, i],fraction=0.046, pad=0.04)
                    ax[0, i].set_title("Velocity {} {} {:.3f}".format(int(50*src[1]), int(50*src[0]), err))

                    im=ax[1, i].imshow(error_im, vmin=0, vmax=3)
                    fig.colorbar(im, ax=ax[1, i],fraction=0.046, pad=0.04)
                    ax[1, i].set_title("Velocity Error {} {}".format(int(50*src[1]), int(50*src[0])))

                plt.tight_layout()
                plt.savefig("{}/FwdModelRecon{}_{}.png".format(args.PATH,str(k).zfill(5),str(k_sub).zfill(5)))
                plt.close()
                
                # SCATTER PLOT RECONSTRUCTED SOURCES             
                img, logdet = GForward(z_sample, GNet, args.nsrc, d, logscale_factor, 
                                       eiko=args.EIKO, xtrue = Xsrc_ext1024, device=args.device,
                                       use_dataparallel=args.use_dataparallel)
                scatter_im = img.detach().cpu().numpy()
                mean_img = np.mean(scatter_im, axis=0)

                PlotScatterAll(args.nsrc, args.nrec, scatter_im, Xrec.detach().cpu().numpy(), X0, mean_img, 
                               gauss_means.detach().cpu().numpy(), 
                               args.px_randmean, args.prior_sigma, alpha=0.1, k=k, k_sub=k_sub, 
                               plot_otherpts=True, path=args.PATH)
                np.save("{}/Data/SrcRecon.npy".format(args.PATH), scatter_im)
                
                fig, ax = plt.subplots()        
                if args.phi_weight > 1e-12:
                    plt.plot(np.log10(loss_phi_list), ":", label = "p(phi)")
                if args.invar_weight is not None:
                    plt.plot(np.log10(loss_invar_list), ":", label= "Velocity Invariance as \n Source Invariance")
                if args.ttinvar is not None:
                    plt.plot(np.log10(loss_ttinvar_list), ":", label = "TT Invariance" )
                plt.plot(np.log10(loss_list), label= "All")
                plt.plot(np.log10(loss_mse_list), ":", label="MSE")
                plt.plot(np.log10(loss_prior_list), ":", label="p(x)")
                plt.legend()
                plt.savefig("{}/loss.png".format(args.PATH))
                plt.close()
                
                
                fig, ax = plt.subplots()        
                if args.EIKO == False:
                    plt.plot(np.log10(loss_prior_list)/args.nsrc, ":", label="p(x)")
                    plt.plot(np.log10(loss_mse_list)/args.nsrc/args.nrec, "--", label = "MSE")
                plt.legend()
                plt.savefig("{}/NormalizedLoss.png".format(args.PATH))
                plt.close()
                
                    
#                 Plot mean velo recon and each recon per receiver
                generate_mean_velo(Xrec.detach().cpu().numpy(), FNet, args.device, 
                                   VTrue=true_velocity_model, use_dataparallel=args.use_dataparallel,
                                   s=51, path=args.PATH, k=k, close=True)
                
            # output log to terminal
            if ((k_sub%args.print_every == 0) and args.EMFull) or ((k%args.print_every == 0) and not args.EMFull):
                if args.ttinvar is not None:
                    print("epoch: {} {}, Loss: {:.8f}, TT Invar Loss: {:.8f}".format(k, k_sub, 
                                                                                          loss_list[-1], loss_ttinvar_list[-1]))
                else:
                    print("epoch: {} {}, Loss: {:.8f}".format(k, k_sub, loss_list[-1]))
                
    fig, ax = plt.subplots()        
    if args.phi_weight > 1e-12:
        plt.plot(np.log10(loss_phi_list), ":", label = "p(phi)")
    
    if args.invar_weight is not None:
        plt.plot(np.log10(loss_invar_list), ":", label= "Velocity Invariance as \n Source Invariance")
    if args.ttinvar is not None:
        plt.plot(np.log10(loss_ttinvar_list), ":", label = "TT Invariance" )
    plt.plot(np.log10(loss_list), label= "All")
    plt.plot(np.log10(loss_mse_list), ":", label="MSE")
    plt.plot(np.log10(loss_prior_list), ":", label="p(x)")

    plt.legend()

    plt.savefig("{}/loss.png".format(args.PATH))
    plt.close()
    
    ############################################# GENERATE FIGURES ###########################################################

    img, logdet = GForward(z_sample, GNet, args.nsrc, d, logscale_factor, 
                           eiko=args.EIKO, xtrue = Xsrc_ext1024, device=args.device,
                           use_dataparallel=args.use_dataparallel)
    image = img.detach().cpu().numpy()


    fig, ax = plt.subplots(2, 4, figsize=(14, 8))
    for i in range(0, 4):
        if args.pltRec == True:
            src = plotRec[i, :]
        else:
            src = np.array([0.25*i, 0.25*i])
        V, TT = VeloRecon(FNet, args.device, num = s, src=src, use_dataparallel=args.use_dataparallel)
        V = V.detach().cpu().numpy().reshape([s, s])
        TT = TT.detach().cpu().numpy().reshape([s, s])
        error_im = np.abs(V.transpose()-true_velocity_model)
        err = np.sum(error_im)/51/51
        
        np.save("{}/Data/VFinalRecon{}.npy".format(args.PATH, i), V)

        im=ax[0, i].imshow(V.transpose(), vmin=0, vmax=10)
        fig.colorbar(im, ax=ax[0, i],fraction=0.046, pad=0.04)
        ax[0, i].set_title("Velocity {} {} {:.3f}".format(int(50*src[1]), int(50*src[0]),err))
        
        im=ax[1, i].imshow(np.abs(V.transpose()-true_velocity_model), vmin=0, vmax=3)
        fig.colorbar(im, ax=ax[1, i],fraction=0.046, pad=0.04)
        ax[1, i].set_title("Velocity Error {} {}".format(int(50*src[1]), int(50*src[0])))
        
    plt.tight_layout()
    plt.savefig("{}/FwdModelReconFinal.png".format(args.PATH))
    plt.close()

    ## SCATTER PLOT RECONSTRUCTED SOURCES 
    scatter_im = image.reshape([n_sample*args.nsrc, 2])
    fig, ax = plt.subplots( figsize=(6, 6))
    if args.nrec < 20:
        plt.scatter(Xrec.detach().cpu().numpy()[:, 0], Xrec.detach().cpu().numpy()[:, 1], c="g")
    plt.scatter(X0[:, 0], X0[:, 1], s=100, c="orange")
    plt.scatter(scatter_im[:, 0], scatter_im[:, 1])
    if args.nrec < 20:
        plt.legend(["Receivers", "True Sources", "Recon Points", "Mean Recon Source"])
    else:
        plt.legend(["True Sources", "Recon Points", "Mean Recon Source"])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig("{}/SourceReconScatter.png".format(args.PATH))
    plt.close()
    

    ## GENERATE SAMPLES Y
    XrecIn = torch.cat(n_sample*[torch.unsqueeze(Xrec, axis=0)])
    y = FForward(img, XrecIn, FNet, args.data_sigma, fwd_velocity, args.device)
    image_out = y.detach().cpu().numpy()

    ## PLOT MEAN STDEV Y
    mean_blur_img = np.mean(image_out, axis=0)
    stdev_blur_img = np.std(image_out, axis=0)
    std_max = stdev_blur_img.max()
    fig, ax = plt.subplots(1, 6, figsize = (15, 4))
    im0=ax[0].imshow(fwd_tt.detach().cpu().numpy().transpose(), vmin=0, vmax=0.3)
    im1=ax[1].imshow(true_tt.detach().cpu().numpy().transpose(), vmin=0, vmax=0.3)
    im2=ax[2].imshow(mean_blur_img.transpose(), vmin=0, vmax=0.3)
    im3=ax[3].imshow(stdev_blur_img.transpose(), vmin=0, vmax=std_max)
    im4=ax[4].imshow(np.abs(mean_blur_img - true_tt.detach().cpu().numpy()).transpose(), vmin=0)
    im5=ax[5].imshow(np.abs(mean_blur_img - true_tt.detach().cpu().numpy()).transpose()/stdev_blur_img.transpose(), vmin=0, vmax=3)

    im = [im0, im1, im2, im3, im4, im5]
    for i in range(6):
        fig.colorbar(im[i], ax=ax[i],fraction=0.046, pad=0.04)
        ax[i].set_xlabel("sources")
        ax[i].set_ylabel("receivers")
    ax[0].set_title("incorrect travel times")
    ax[1].set_title("true travel times")
    ax[2].set_title("mean travel times")
    ax[4].set_title("mean error")
    ax[3].set_title("image stdev")
    ax[5].set_title("mean error/stdev error")
    plt.tight_layout()

    plt.savefig("{}/MeanStdevTT.png".format(args.PATH))
    plt.close()
    
    if args.EIKO == True:
        generate_gif(args.PATH, "VelocityRecon", args.num_epochs, args.num_subepochsM,  args.save_every, 1)
        generate_gif(args.PATH, "FwdModelRecon", args.num_epochs, args.num_subepochsM,  args.save_every, 1)
        generate_gif(args.PATH, "SourceReconScatter", args.num_epochs, args.num_subepochsM, args.save_every, 1)

        generate_gif(args.PATH, "VelocityRecon", args.num_epochs, args.num_subepochsM, 1, args.save_every)
        generate_gif(args.PATH, "FwdModelRecon", args.num_epochs, args.num_subepochsE, 1, args.save_every)
        generate_gif(args.PATH, "SourceReconScatter", args.num_epochs, args.num_subepochsE, 1, args.save_every)
    
    #   PLOT LOSSES
    fig, ax = plt.subplots()
    ax.plot(np.log10(tt_err_list)[-100:])
    ax.plot(np.log10(tt_true_err_list)[-100:])
    ax.plot(np.log10(source_err_list)[-100:])

    ax.legend(["TT MSE", "TT True Error", "Source Error"])
    plt.savefig("{}/SVTrunc_error.png".format(args.PATH))

    # PLOT LOSS
    fig, ax = plt.subplots()
    ax.plot(np.log10(tt_err_list)[-100:])
    ax.plot(np.log10(velo_err_list)[-100:])
    ax.plot(np.log10(tt_true_err_list)[-100:])
    ax.legend(["TT MSE", "Velo MSE", "TT True Error"])
    plt.savefig("{}/TTVTrunc_error.png".format(args.PATH))



if __name__ == "__main__":
      
    parser = argparse.ArgumentParser(description='EikoNet with Residual')
    parser.add_argument('--btsize', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 752)')
    parser.add_argument('--num_epochs', type=int, default=3500, metavar='N',
                        help='number of epochs to train (default: 3500)')
    parser.add_argument('--num_subepochsE', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 400)')
    parser.add_argument('--num_subepochsM', type=int, default=600, metavar='N',
                        help='number of epochs to train (default: 600)')
    parser.add_argument('--save_every', type=int, default=500, metavar='N',
                        help='checkpoint model (default: 100)')
    parser.add_argument('--save_everyE', type=int, default=500, metavar='N',
                        help='checkpoint model for E step (default: 100)')
    parser.add_argument('--print_every', type=int, default=50, metavar='N',
                        help='checkpoint model (default: 50)')
    parser.add_argument('-dir', '--dir', type=str, default="Test",
                        help='output folder')
    
    
    parser.add_argument('--dv', type=str, default='cuda:1',
                        help='enables CUDA training')
    parser.add_argument('--multidv', type=int, nargs='+', default = None,
                       help="use multiple gpus (default: None for all) use -1 for just the dv device")
    
    
    # Training method configurations
    parser.add_argument('--DPI', action='store_true', default=False, 
                        help='use DPI (true forward)')
    parser.add_argument('--EIKO', action='store_true', default=False, 
                        help='use eikonet method assuming true soruces(default: False )')
    parser.add_argument('--sampled', action='store_true', default=False, 
                        help='use Eiko with sampled sources (default: False )')
    parser.add_argument('--large', action='store_true', default=False, 
                        help='use eikonet8 for architecture (default: False)')
    parser.add_argument('--EMFull', action='store_true', default=False, 
                        help='True: E to convergence, M to convergence False: alternate E, M every epoch (default: False)')
    
    
    # network setup
    parser.add_argument('--load', action='store_true', default=False,
                        help='output folder')
    parser.add_argument('--randinit', action='store_true', default=False,
                        help='output folder')
#     parser.add_argument('--GInit', action='store_true', default=False,
#                         help='initialize Gnetwork')
    parser.add_argument('--use_eikonet', action='store_true', default=False,
                        help='use eikonet')
    parser.add_argument('--use_invareiko', action='store_true', default=False, 
                        help='use source rec invariant EikoNet (true sources)')
    parser.add_argument('--velo_loss', action='store_true', default=False,
                        help='use velo as loss (default: false)')
    parser.add_argument('--sine_activation', action='store_true', default=False,
                        help='use sine instead of ELU')
    parser.add_argument('--ffreqs', type=int, default=None,
                        help='use fourier frequencies model (default None)')
    parser.add_argument('--sine_freqs', type=int, default=1,
                        help='use fourier frequencies model (default None)')
    parser.add_argument('--vnet_invar', action='store_true', default = False, 
                         help="velocity source invariance as network (default False)")
   
    
    
    
    # Source Receiver Configuration
    parser.add_argument('--center', action='store_true', default=False, 
                        help='use source at center when there is only 1 source')
    parser.add_argument('--nsrc', type=int, default=5,
                        help='number of sources')
    parser.add_argument('--nrec', type=int, default=20,
                        help='number of receivers')
    parser.add_argument('--surfaceR', type=int, default=0, 
                        help='number of surfaces: only use receivers at surface')
    parser.add_argument('--pltRec', action='store_true', default=False, 
                        help='plot recon from 4 random receivers')
    
    
    # velocity input
    parser.add_argument('--model', type=str, default="Blur1", 
                        help='use blur model (true forward)')
    parser.add_argument('--fwdmodel', type=str, default=None, 
                        help='use blur model (assumed forward)')
    parser.add_argument('--fwd_velocity', type=float, default=None, 
                        help='assumed forward velocity (default: mean of true model)')
    parser.add_argument('--init_prior', action='store_true', default = False, 
                         help="initialize with a model learned from incorrect sources sampled from prior (default False)")
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--Elr', type=float, default=1e-4, 
                        help='learning rate(default: 1e-4)')
    parser.add_argument('--Mlr', type=float, default=1e-6, 
                        help='learning rate(default: 1e-6)')
    
    # source receiver config
    
    parser.add_argument('--px_randmean', action='store_true', default=False,
                       help='use random mean for gaussian prior')
    parser.add_argument('--gridsrcs', action='store_true', default=False,
                       help='use grid for sources')
    
    # Loss function parameters
    parser.add_argument('--prior_sigma', type=float, default=1e-1, 
                        help='prior sigma (default: 1e-3)')
    parser.add_argument('--prior_weight', type=float, default=1, 
                        help='prior weight (default: 1)')
    
    parser.add_argument('--data_sigma', type=float, default=2.5e-2, 
                    help='data sigma (default: 1e-3)')
    parser.add_argument('--data_weight', type=float, default=1, 
                    help='data weight (default: 1)')
    parser.add_argument('--nonoise', action="store_true", default=False, 
                         help="no measurement noise (default False)")
    parser.add_argument('--reduction', type=str, default="mean", 
                         help="use SSE instead of MSE (default 'mean', alt 'sum')")
    
    parser.add_argument('--phi_weight', type=float, default=1e-3, 
                    help='weight on prior function (default: 1e-3)')
    
    
    parser.add_argument('--invar_weight', type=float, default=None, 
                         help="velocity source invariance (as loss) weight (default 0)")
    parser.add_argument('--scheduled_vinvar', action="store_true", default=False, 
                         help="schedule vinvar weight (default False)")
    parser.add_argument('--vnet_weight', type=float, default=1e-6, 
                         help="velocity source invariance (as network) weight (default 1e-6)")
    parser.add_argument('--invar_rec', action="store_false", default=True, 
                         help="use random receiver for velocity invariance (default True)")
    parser.add_argument('--ttinvar', type=float, default=None, 
                         help="tt symmetry weight (default None)")

    args = parser.parse_args()    
    
    args.PATH = "SeismoGEMResults/MAP/{}_nsrc{}_nrec{}".format(args.dir, args.nsrc, args.nrec)
        
#     if torch.cuda.is_available():
#         args.device = args.dv
#     else:
#         args.device = 'cpu'

    if torch.cuda.is_available():
        args.device = args.dv
        dv = int(args.device[-1])
        if args.EIKO == False:
            if args.multidv is None:
                arr = [i for i in range(torch.cuda.device_count())]
                args.device_ids = [dv] + arr[0:dv] + arr[dv+1:]
            elif args.multidv == [-1]:
                args.device_ids = [dv]
            else:
                args.device_ids = [dv] + args.multidv
    else:
        args.device = 'cpu'  
    print(args.device_ids)
        
    args.use_dataparallel = False 
    if torch.cuda.device_count() > 1 and args.EIKO == False:
        args.use_dataparallel = True
        

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(args)
      
        
    USE_GPU = True

#     dtype = torch.double # we will be using double throughout this tutorial
    print("cuda available ", torch.cuda.is_available())
    print("---> num gpu", torch.cuda.device_count())
    print('---> using device:', args.device)

    args.num_epochs = args.num_epochs+1
    args.num_subepochsE = 1
    args.num_subepochsM = 1
    print("Stochastic EM w/ {} epochs and {} E subepochs {} M subepochs".format(args.num_epochs, 
                                                                              args.num_subepochsE, args.num_subepochsM))
    
#     if args.EIKO == True:
#         n = args.num_epochs
#     else:
#         if args.EMFull == True:
#             n = args.num_subepochsE
#         else:
#             n = args.num_epochs

       
    
    ############################## CREATE DIRECTORIES ###########################################
    try:
        # Create target Directory
        os.mkdir(args.PATH)
        print("Directory " , args.PATH ,  " Created ") 
    except FileExistsError:
        print("Directory " , args.PATH ,  " already exists")
    try:
        # Create target Directory
        os.mkdir(args.PATH+"/Data")
        print("Directory " , args.PATH+"/Data" ,  " Created ") 
    except FileExistsError:
        print("Directory " , args.PATH+"/Data" ,  " already exists")
    with open("{}/args.json".format(args.PATH), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    
    main_function(args)



