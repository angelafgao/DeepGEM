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
    n_flow = 16
    samples = 51
    logdet = 1
    affine = True
   
    use_bias = False

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

        Xrec_idx = np.random.choice(idx, [args.nrec, d]).astype(np.int)
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
            Xsrc_idx = (np.random.choice(idx, [args.nsrc, d])*0 + samples//2).astype(np.int)
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
                Xsrc_idx = np.random.choice(idx, [args.nsrc, d]).astype(np.int)
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

        
    print("Using Eikonet()")
    FNet = EikoNet(input_size = d, sine_activation=args.sine_activation, sine_freq = args.sine_freqs)
    FNet.init_model(args.fwdmodel, args.randinit, args.device, init_prior=False)

    if args.use_dataparallel == True:
        print("Parallel Training with {} GPUS".format(len(args.device_ids)))
        GNet = nn.DataParallel(GNet, device_ids = args.device_ids)
        GNet.to(args.device)
        FNet = nn.DataParallel(FNet, device_ids = args.device_ids)
        FNet.to(args.device) 

    VNet = None
        
    FNet.to(args.device)

    # Load Forward Model Method
    if args.fwdmodel is None:
        FTrue = lambda Xrec, Xsrc, v: generate_tt_homogeneous(Xrec, Xsrc, v, use_torch=True)
    else:
        output_matrix = np.load("SeismoData/{}_XTTPairs.npy".format(args.fwdmodel))
        FTrue = lambda idx: Tensor(output_matrix[idx, -1][np.newaxis, :, np.newaxis]).to(args.device)

    flux = np.sum(Xsrc.cpu().numpy())

    criterion=nn.MSELoss(reduction=args.reduction)

    logscale_factor = Img_logscale(scale=flux/(0.8*d*args.nsrc)).to(args.device)
    
    Xsrc_ext = torch.cat(args.btsize*[torch.unsqueeze(Xsrc, axis=0)], axis=0)
    if args.EIKO == True:
        n_sample = 2
    else:
        n_sample = 1024#1024#1024#
    Xsrc_ext1024 = torch.cat(n_sample*[torch.unsqueeze(Xsrc, axis=0)], axis=0)
    X0 = Xsrc.detach().cpu().numpy()

    if args.px_randmean == False:
        print("Using True Means for Source Location Prior")
        prior_gauss = lambda x: criterion(x, Xsrc_ext)/args.prior_sigma**2
    else:
        print("Using Random Means for Source Location Prior")
        gauss_means_ext = torch.cat(args.btsize*[torch.unsqueeze(gauss_means, axis=0)], axis=0)
        prior_gauss = lambda x: criterion(x, gauss_means_ext)/args.prior_sigma**2
    prior_unif = lambda x: torch.sum((x < 1)*(x > 0))

    prior = prior_gauss
    print("Gaussian Prior on Xsrc", prior_gauss(Xsrc_ext))

    ### DEFINE OPTIMIZERS 
    Eoptimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,list(GNet.parameters())
                                         +list(logscale_factor.parameters())),lr = args.Elr)

    Moptimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,list(FNet.parameters()) ), lr = args.Mlr)

    
    #################################### TRAINING #########################################################

    Eloss_list = []
    Eloss_prior_list = []
    Eloss_mse_list = []
    Eloss_q_list = []

    Mloss_list = []
    Mloss_mse_list = []
    Mloss_phi_list = []
    Mloss_invar_list = []
    Mloss_netinvar_list = []
    Mloss_ttinvar_list = []
    
    velo_err_list = []
    tt_err_list = []
    tt_true_err_list = []
    source_err_list = []

    z_sample = torch.randn(args.btsize, 2*args.nsrc).to(device=args.device)
    x_sample = torch.randn(args.btsize, 2*args.nsrc).to(device=args.device)
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
    
    diversity_weight_arr = []
    vinvar_weight_arr = []
    ttinvar_weight_arr = []
    diversity_weight = 1 
    
    sample_noise = args.prior_sigma*torch.randn(Xsrc_ext.shape).to(args.device)
    if args.sampled == True:
        np.save("{}/Data/XsrcWithNoise.npy".format(args.PATH), (sample_noise + Xsrc_ext).detach().cpu().numpy())
    
    if args.invar_weight is not None:
#         if args.scheduled_vinvar == True:
#             # steps
#             vinvar_weight_diff = (-5+np.log10(args.invar_weight))
#             vinvar_weight = 10**(np.ceil(vinvar_weight_diff))
#         else:
        vinvar_weight = args.invar_weight

    else:
        vinvar_weight = 0
    if args.ttinvar is not None:
        ttinvar_weight = args.ttinvar
    else:
        ttinvar_weight = 0
   
    for k in range(args.num_epochs):
        
        ########## E STEP Update Generator Network ##############
        if args.EIKO == False and (k == 0 and args.EMFull == True) :
            diversity_weight = 1   
            print("Reset diversity")

            if args.invar_weight is not None:
#                 if args.scheduled_vinvar == True:
#                     # steps
#                     vinvar_weight_diff = (-5+np.log10(args.invar_weight))
#                     vinvar_weight = 10**(np.ceil(vinvar_weight_diff))
#                 else:
                vinvar_weight = args.invar_weight
    
            else:
                vinvar_weight = 0
            if args.ttinvar is not None:
                ttinvar_weight = args.ttinvar
            else:
                ttinvar_weight = 0
            
        diversity_weight_arr.append(diversity_weight)
        vinvar_weight_arr.append(vinvar_weight)
        ttinvar_weight_arr.append(ttinvar_weight)
        logdet_weight = diversity_weight
        
        diversity_weight = 1

        # SET UP NUMBER OF E GRADIENT UPDATES 
        if args.EMFull == False and args.EIKO == False and k == 0:
            num_subepochsE = 800 # stochstic EM run 1 full E pass
            print("FULL E PASS")
        elif args.EMFull == True and k == 0:
            num_subepochsE = 0 # no E step for the first round of full EM
            print("NO 0 E STEP")
        else:
            num_subepochsE = args.num_subepochsE
            
        for k_sub in range(num_subepochsE):
            if args.EIKO == False:
                z_sample = torch.randn(args.btsize, 2*args.nsrc).to(device=args.device)
                if args.EIKO == False and k == 0:
                    data_weight = 0
                    beta = 1
                else:
                    data_weight = 1/args.data_sigma**2
                    beta = logdet_weight
                
                Eloss, qloss, priorloss, mseloss = EStep(z_sample, XrecIn, args.device, true_tt_ext, 
                                                         GNet, FNet, prior, 
                                                         data_weight, args.prior_weight, args.prior_sigma, 
                                                         args.data_sigma, velocity=fwd_velocity,
                                                         nsrc = args.nsrc, d=d, logscale_factor=logscale_factor,
                                                         logdet_weight=beta, velo_loss=args.velo_loss,
                                                         use_dataparallel = args.use_dataparallel, reduction=args.reduction)
                
                Eloss_list.append(Eloss.detach().cpu().numpy())
                Eloss_prior_list.append(priorloss.detach().cpu().numpy())
                Eloss_q_list.append(qloss.detach().cpu().numpy())
                Eloss_mse_list.append(mseloss.detach().cpu().numpy())
                Eoptimizer.zero_grad()
                Eloss.backward()
                nn.utils.clip_grad_norm_(list(GNet.parameters())+ list(logscale_factor.parameters()), 10)
                Eoptimizer.step()
                
                ## SCATTER PLOT RECONSTRUCTED SOURCES             
                z_sample = torch.randn(n_sample, 2*args.nsrc).to(device=args.device)
                img, logdet = GForward(z_sample, GNet, args.nsrc, d, logscale_factor, 
                                       eiko=args.EIKO, xtrue = Xsrc_ext1024, device=args.device,
                                       use_dataparallel=args.use_dataparallel)
                source_err = nn.MSELoss()(img, Xsrc_ext1024)
                source_err_list.append(source_err.item())
                del img, logdet, source_err, z_sample
                
                # CHECKPOINT    
                if ((k_sub%args.save_everyE == 0) and args.EMFull) or ((k != 0) and (k%args.save_everyE == 0) and not args.EMFull):
                    

                    if args.EIKO == False:
                        with torch.no_grad():
                            if args.use_dataparallel == True:
                                torch.save({
                                'epoch':k,
                                'model_state_dict': GNet.module.state_dict(),
                                'optimizer_state_dict': Eoptimizer.state_dict(),
                                }, '{}/{}{}_{}.pt'.format(args.PATH,"GeneratorNetwork",str(k).zfill(5),str(k_sub).zfill(5)))
                            else:
                                torch.save({
                                    'epoch':k,
                                    'model_state_dict': GNet.state_dict(),
                                    'optimizer_state_dict': Eoptimizer.state_dict(),
                                    }, '{}/{}{}_{}.pt'.format(args.PATH,"GeneratorNetwork",str(k).zfill(5),str(k_sub).zfill(5)))

                    ## SCATTER PLOT RECONSTRUCTED SOURCES             
                    z_sample = torch.randn(n_sample, 2*args.nsrc).to(device=args.device)
                    img, logdet = GForward(z_sample, GNet, args.nsrc, d, logscale_factor, 
                                           eiko=args.EIKO, xtrue = Xsrc_ext1024, device=args.device,
                                           use_dataparallel=args.use_dataparallel)
                    scatter_im = img.detach().cpu().numpy()
                    mean_img = np.mean(scatter_im, axis=0)

                    PlotScatterAll(args.nsrc, args.nrec, scatter_im, Xrec.detach().cpu().numpy(), X0, mean_img, 
                                   gauss_means.detach().cpu().numpy(), 
                                   args.px_randmean, args.prior_sigma, alpha=0.1, k=k, k_sub=k_sub, 
                                   plot_otherpts=True, path=args.PATH)
                    
                    GenerateDensityMap(nsrc=args.nsrc, Xsrc=X0, scatter_im=scatter_im,  gauss_means= gauss_means.detach().cpu().numpy(),
                                       scale=1/0.04/n_sample, bins=201, path=args.PATH, k=k, k_sub=k_sub)



                if ((k_sub%args.print_every == 0) and args.EMFull) or ((k%args.print_every == 0) and not args.EMFull):
                    if args.EIKO == False:
                        print("epoch: {} {}, E step loss: {:.8f}".format(k, k_sub,Eloss_list[-1]) )
                
        ########## M STEP Update Eiko Network ##############
        
        # RUN 1 M STEP FOR EM
        if args.EIKO == False and k == 0:
            if args.prior_sigma < 1:
                if args.init_prior == True:
                    sampled = False
                    print("RUN M STEP 0, sampled from prior data")
                else:
                    sampled = True
                    print("RUN M STEP 0, sampled from prior, noisy data")
                eiko = True
                xtrue = gauss_means_ext
    #             xtrue = Xsrc_ext
    #             data_in = true_tt_nonoise_ext
                data_in = true_tt_ext
                num_subepochsM = 1001
            else:
                num_subepochsM = 0
                print("SKIP M STEP 0 because prior is too big")
        else: # original parameters
            sampled = args.sampled
            eiko=args.EIKO
            if args.init_prior == True:
                xtrue = gauss_means_ext
            else:
                xtrue = Xsrc_ext
            data_in = true_tt_ext
            num_subepochsM = args.num_subepochsM
            
        for k_sub in range(num_subepochsM):
            if args.EIKO == True:
                x = k
            else:
                x = k_sub
            if args.invar_weight is not None:
#                 if args.scheduled_vinvar == True:
#                     vinvar_weight_diff = x*args.vinvar_alpha + (-5+np.log10(args.invar_weight))
#                     vinvar_weight = 10**(np.ceil(vinvar_weight_diff))
#                 else:
                vinvar_weight = args.invar_weight
            else:
                vinvar_weight = 0
            if args.ttinvar is not None:
                ttinvar_weight = args.ttinvar # constant
            else:
                ttinvar_weight = 0
            
            
            z_sample = torch.randn(args.btsize, 2*args.nsrc).to(device=args.device)
        
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


            Mloss, philoss, mse, invar, netinvar, ttinvarloss = MStep(z_sample, XrecIn, x_sample_src, x_sample_rec, 
                                                                      args.device, data_in, GNet, FNet,
                                                                      args.phi_weight, FTrue, args.data_sigma, velocity=fwd_velocity, 
                                                                      fwd_velocity=fwd_velocity, nsrc = args.nsrc, d=d, 
                                                                      logscale_factor=logscale_factor, eiko=eiko, 
                                                                      xtrue = Xsrc_ext, fwdmodel = args.fwdmodel, 
                                                                      xidx = x_idx, velo_loss=args.velo_loss, 
                                                                      invar_weight = vinvar_weight, 
                                                                      invar_src = invar_src, invar_rec = invar_rec, VNet=VNet, 
                                                                      vnet_weight = args.vnet_weight, ttinvar = ttinvar_weight,
                                                                      use_dataparallel = args.use_dataparallel, 
                                                                      fwd_velocity_model = fwd_velocity_model, 
                                                                      sampled=sampled, prior_x = args.prior_sigma,
                                                                      samplenoise = None)

            Mloss_list.append(Mloss.detach().cpu().numpy())
            Mloss_mse_list.append(mse.detach().cpu().numpy())
            Mloss_phi_list.append(philoss)
            Mloss_invar_list.append(invar)
            Mloss_netinvar_list.append(netinvar)
            Mloss_ttinvar_list.append(ttinvarloss)
            Moptimizer.zero_grad()
            Mloss.backward()
            nn.utils.clip_grad_norm_(list(FNet.parameters()), 1)
            Moptimizer.step()

            
            # CHECKPOINT
            if ((k_sub%args.save_every == 0) and args.EMFull) or ((k%args.save_every == 0) and  not args.EMFull):
                with torch.no_grad():
                    if args.use_dataparallel == True:
                        torch.save({
                            'epoch':k,
                            'model_state_dict': FNet.module.state_dict(),
                            'optimizer_state_dict': Moptimizer.state_dict(),
                            }, '{}/{}{}_{}.pt'.format(args.PATH,"ForwardNetwork",str(k).zfill(5),str(k_sub).zfill(5)))
                    else:
                        torch.save({
                            'epoch':k,
                            'model_state_dict': FNet.state_dict(),
                            'optimizer_state_dict': Moptimizer.state_dict(),
                            }, '{}/{}{}_{}.pt'.format(args.PATH,"ForwardNetwork",str(k).zfill(5),str(k_sub).zfill(5)))


                if args.EIKO == False:
                    fig, ax = plt.subplots(1, 2, figsize = (15, 4))
                    ax[0].plot(np.log10(Eloss_list))
                    ax[0].plot(np.log10(Eloss_prior_list), ":")
                    ax[0].plot(np.log10(Eloss_mse_list), "--")
                    ax[0].plot(np.log10(Eloss_q_list), ":")
                    ax[0].legend(["Estep", "p(x)", "E step mse", "q"])
                    ax[1].plot(np.log10(Mloss_list))
                    ax[1].plot(np.log10(Mloss_phi_list), ":")
                    ax[1].plot(np.log10(Mloss_mse_list), ":")

                    ax[1].plot(np.log10(Mloss_list), label= "M All")
                    if args.phi_weight > 1e-12:
                        ax[1].plot(np.log10(Mloss_phi_list), ":", label = "p(phi)")
                    ax[1].plot(np.log10(Mloss_mse_list), ":", label="Mstep MSE")
                    if args.invar_weight > 1e-12:
                        ax[1].plot(np.log10(Mloss_invar_list), ":", label= "Velocity Invariance as \n Source Invariance")
                    if args.ttinvar is not None:
                        ax[1].plot(np.log10(Mloss_ttinvar_list), ":", label = "TT Invariance" )

                    ax[1].legend()
                    plt.savefig("{}/SeparatedLoss.png".format(args.PATH), format='svg', bbox_inches = 'tight', pad_inches = 0)
                    plt.close()
                    
#                 Plot mean velo recon and each recon per receiver
                generate_mean_velo(Xrec.detach().cpu().numpy(), FNet, args.device, 
                                   VTrue=true_velocity_model, use_dataparallel=args.use_dataparallel,
                                   s=51, path=args.PATH, k=k, close=True)
                
            # output log to terminal
            if ((k_sub%args.print_every == 0) and args.EMFull) or ((k%args.print_every == 0) and not args.EMFull):
                if args.ttinvar is not None:
                    print("epoch: {} {}, M Loss: {:.8f}, TT Invar Loss: {:.8f}".format(k, k_sub, 
                                                                                          Mloss_list[-1], Mloss_ttinvar_list[-1]))
                else:
                    print("epoch: {} {}, M Loss: {:.8f}".format(k, k_sub, Mloss_list[-1]))

if __name__ == "__main__":
      
    parser = argparse.ArgumentParser(description='EikoNet with Residual')
    parser.add_argument('--btsize', type=int, default=128, metavar='N',
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
#     parser.add_argument('--DPI', action='store_true', default=False, 
#                         help='use DPI (true forward)')
    parser.add_argument('--EIKO', action='store_true', default=False, 
                        help='use eikonet method assuming true soruces(default: False )')
    parser.add_argument('--sampled', action='store_true', default=False, 
                        help='use Eiko with sampled sources (default: False )')
#     parser.add_argument('--large', action='store_true', default=False, 
#                         help='use eikonet8 for architecture (default: False)')
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
#     parser.add_argument('--use_invareiko', action='store_true', default=False, 
#                         help='use source rec invariant EikoNet (true sources)')
    parser.add_argument('--velo_loss', action='store_true', default=False,
                        help='use velo as loss (default: false)')
    parser.add_argument('--sine_activation', action='store_true', default=False,
                        help='use sine instead of ELU')
#     parser.add_argument('--ffreqs', type=int, default=None,
#                         help='use fourier frequencies model (default None)')
    parser.add_argument('--sine_freqs', type=int, default=1,
                        help='use fourier frequencies model (default None)')
    
 
    
    # Source Receiver Configuration
    parser.add_argument('--center', action='store_true', default=False, 
                        help='use source at center when there is only 1 source')
    parser.add_argument('--nsrc', type=int, default=5,
                        help='number of sources')
    parser.add_argument('--nrec', type=int, default=20,
                        help='number of receivers')
    parser.add_argument('--surfaceR', type=int, default=0, 
                        help='number of surfaces: only use receivers at surface')
#     parser.add_argument('--pltRec', action='store_true', default=False, 
#                         help='plot recon from 4 random receivers')
    
    
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
#     parser.add_argument('--data_weight', type=float, default=1, 
#                     help='data weight (default: 1)')
    parser.add_argument('--nonoise', action="store_true", default=False, 
                         help="no measurement noise (default False)")
    parser.add_argument('--reduction', type=str, default="sum", 
                         help="use SSE instead of MSE (default 'sum', alt 'mean')")
    
    parser.add_argument('--phi_weight', type=float, default=1e-3, 
                    help='weight on prior function (default: 1e-3)')
    
#     parser.add_argument('--beta', type=float, default=1, 
#                          help="use beta term for decaying entropy")
#     parser.add_argument('--base', type=float, default=np.e, 
#                          help="base of exponential for decaying function")
    
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
    
    args.PATH = "SeismoGEMResults/EM/{}_nsrc{}_nrec{}".format(args.dir, args.nsrc, args.nrec)
        

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
        
    args.use_dataparallel = False 
    if torch.cuda.device_count() > 1 and args.EIKO == False:
        args.use_dataparallel = True
        
    if args.velo_loss == True:
        args.btsize = 1
    
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
    
    if args.EMFull == True:
        args.num_epochs = args.num_epochs+1
        args.num_subepochsE = args.num_subepochsE+1
        args.num_subepochsM = args.num_subepochsM+1
        print("Full EM w/ {} epochs and {} E subepochs {} M subepochs".format(args.num_epochs, args.num_subepochsE, args.num_subepochsM))
    else:
        args.num_epochs = args.num_epochs+1
        args.num_subepochsE = 1
        args.num_subepochsM = 1
        print("Stochastic EM w/ {} epochs and {} E subepochs {} M subepochs".format(args.num_epochs, 
                                                                                  args.num_subepochsE, args.num_subepochsM))
    args.alpha=0
    print("Entropy decay function beta {} alpha {}".format(args.beta, args.alpha))

    
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



