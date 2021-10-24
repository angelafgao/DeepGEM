#!/usr/bin/env python
# coding: utf-8

import psutil
# psutil.Process().nice(1)

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
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler,WeightedRandomSampler
print(torch.__version__)

from scipy.signal import convolve2d
print(scipy.__version__)
from generative_model import realnvpfc_model
from generative_model import glow_model

from deconv_helpers import *
import argparse



def main_function(args):

    n_flow = 16
    n_block = args.n_block # GLOW
    npix = args.npix
    l1 = 0
    n_channel = 1 #GLOW
    no_lu = False #GLOW

    Elr = 4e-4
    Mlr = 3e-4
    affine = True #Both GLOW and RealNVP
    data_weight = 1/args.data_sigma**2
    
    if npix > 64:
        seqfrac = 4/(64**args.seqfracpow)*(npix**args.seqfracpow)
    else:
        seqfrac = 4
        
    ################################################ SET UP DATA ####################################################
    

    if args.image == "Milan":
        img_true = np.load("DeconvData/img_milan.npy")
#         img_blur = np.load("img_milan_blur.npy")
    elif args.image == "Annecy":
        img_true = np.load("DeconvData/img_annecy.npy")
#         img_blur = np.load("img_annecy_blur.npy")
    elif args.image == "DeadLeaf":
        img_true = np.load("DeconvData/img_true.npy")
#         img_blur = np.load("img_blur.npy")
    else:
        img_true = np.load("DeconvData/{}.npy".format(args.image))
    
    if args.kernel == None:
        kernel = np.load("DeconvData/kernel.npy")
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 0, 1]])/7

        kernel_fwd = np.array([[1, 0, 1],
                              [1, 0, 1],
                              [1, 0, 1]])/6
    else:
        kernel = np.load("DeconvData/{}.npy".format(args.kernel))
        mask = np.load("DeconvData/{}_mask.npy".format(args.kernel))
        kernel_fwd = kernel*mask
        kernel_fwd = kernel_fwd/np.sum(kernel_fwd)
    
    ConvLayer = torch.nn.Conv2d(1, 1, kernel.shape, padding=(kernel.shape[0]-1)//2, padding_mode=args.padding_mode, bias=False)
    ConvLayer.weight.data = Tensor([[kernel]])

    ConvLayerFwd = torch.nn.Conv2d(1, 1, kernel_fwd.shape, padding=(kernel_fwd.shape[0]-1)//2, padding_mode=args.padding_mode, bias=False)
    ConvLayerFwd.weight.data = Tensor([[kernel_fwd]])
    
    img_fwd = ConvLayerFwd(Tensor([[img_true]])).detach().cpu().numpy().reshape(img_true.shape)
    img_blur = ConvLayer(Tensor([[img_true]])).detach().cpu().numpy().reshape(img_true.shape)
    np.save("{}/Data/img_true.npy".format(args.PATH), img_fwd)
    np.save("{}/Data/img_blur.npy".format(args.PATH), img_blur)


    ####################################### SET UP FIGURES ###################################################

    fig, ax = plt.subplots(1, 5, figsize = (13, 3))
    ax[0].imshow(img_true, vmin=0, vmax=1)
    ax[1].imshow(img_blur, vmin=0, vmax=1)
    ax[2].imshow(kernel, vmin=0, vmax=1)
    ax[3].imshow(img_fwd, vmin=0, vmax=1)
    ax[4].imshow(kernel_fwd, vmin=0, vmax=1)

    ax[0].set_title("True Image")
    ax[1].set_title("Blurred Image")
    ax[2].set_title("True Kernel")
    ax[3].set_title("Fwd Model Blurred Image")
    ax[4].set_title("Fwd Model Kernel")

    for i in range(5):
        ax[i].axis("off")
    plt.savefig("{}/setup.png".format(args.PATH))
        
    
    ############################################## MODEL SETUP #####################################################

    img_blur = torch.Tensor(img_blur).to(device=args.device)
    img_true = torch.Tensor(img_true).to(device=args.device)
    ConvLayer = ConvLayer.to(args.device)
        
    if args.softplus == True:
        kernel_network =  KNetwork(args.num_layers, layer_size = args.layer_size, softplus = args.softplus,
                                   hardk = args.hard_k, beta=args.softplusbeta, padding_mode = args.padding_mode).to(args.device)
        print("Kernel Network Size ".format(args.layer_size), kernel_network, " with gauss init")
    else:
        kernel_network = KernelNetwork(args.num_layers).to(args.device)
        if args.kernel == None:
            kernel_network.load("DPIResults/KernelNetwork/BlurPretrain/KernelNetwork00950.pt", args.device)
            print("Kernel Network Weird", kernel_network, " with pretrained init")
        else:
            kernel_network.apply(init_kernel_gauss).to(args.device)
            print("Kernel Network", kernel_network, " with gauss init")
            
    # image size (n_channel, npix, npix)        
    if args.model_name == "realnvp":
        if args.reverse == True:
            print("Generating Reverse RealNVP Network")
            permute = 'reverse'
        else:
            print("Generating Random RealNVP Network")
            permute = 'random'
        img_generator = realnvpfc_model.RealNVP(npix*npix, n_flow, seqfrac = seqfrac, affine=affine, permute = permute).to(args.device)
#         img_generator.apply(init_weights)
        z_shapes = None
    elif args.model_name == "glow":
        print("Generating GLOW Network")
        n_channel = 1
        affine = True
        no_lu = False#True
        z_shapes = glow_model.calc_z_shapes(n_channel, npix, n_flow, n_block)
        print(z_shapes)
        img_generator = glow_model.Glow(n_channel, n_flow, n_block, affine=affine, conv_lu=not no_lu).to(args.device)

    print("Models Initialized")
    FTrue = torch.nn.Conv2d(1, 1, kernel_fwd.shape, padding=(kernel_fwd.shape[0]-1)//2, bias=False).to(args.device)
    FTrue.weight.data = Tensor([[kernel_fwd]])
    FTrue.to(args.device)
    
    
    # MULTIPLE GPUS
    if torch.cuda.device_count() > 1:
        img_generator = nn.DataParallel(img_generator, device_ids = args.device_ids)
        img_generator.to(args.device)
        kernel_network = nn.DataParallel(kernel_network, device_ids = args.device_ids)
        kernel_network.to(args.device)    
    
    
    x_sample = torch.randn(128, npix*npix).to(device=args.device)
  
    flux = np.sum(img_true.cpu().numpy())

    logscale_factor = Img_logscale(scale=flux/(0.8*npix*npix)).to(args.device)

    imgl1_weight = l1 / flux
    imgtv_weight = args.px_weight * npix / flux
    logdet_weight = 1/npix/npix

    prior_TV = lambda x, weight: weight*Loss_TV(x) if weight>0 else 0    
    
    # DEFINE ALL OF THE PRIORS
    kerl1 = lambda kernel_network: torch.abs(torch.sum(create_kernel_torch(kernel_network, args.device, args.num_layers)))
    ker_softl1 = lambda kernel_network: torch.abs(1-torch.sum(kernel_network.module.generatekernel()))
    f_phi_prior = lambda kernel: torch.norm(kernel, args.prior_phi)
    print("Kernel Norm", torch.sum(kernel_network.module.generatekernel()))
    print("Using L{} Norm Prior on Kernel ".format(args.prior_phi))
    
    prior = prior_TV

    criterion=nn.MSELoss()

#     print("TV Prior on image", prior(img_true, 1))

    ### DEFINE OPTIMIZERS 
    Eoptimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,list(img_generator.parameters())
                                         +list(logscale_factor.parameters())),lr = args.Elr)
    Moptimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,list(kernel_network.parameters()) ), lr = args.Mlr)

    
    #################################### TRAINING #########################################################

    Eloss_list = []
    Eloss_prior_list = []
    Eloss_mse_list = []
    Eloss_q_list = []

    Mloss_list = []
    Mloss_mse_list = []
    Mloss_phi_list = []
    Mloss_kernorm_list=[]
    Mloss_phiprior_list=[]

    z_sample = generate_sample(args.model_name, 2, npix, args.device, z_shapes)
#     z_sample = torch.randn(2, npix*npix).to(device=args.device)

    img, logdet = GForward(z_sample, img_generator, npix,logscale_factor)
    image = img.detach().cpu().numpy()
    y = FForward(img, kernel_network, args.data_sigma, args.device, args.hard_k, kerl1)
    image_blur = y.detach().cpu().numpy()

    print("Check Initialization", image.max(), image.min(), image_blur.max(), image_blur.min())
        
    if args.softplus == False:
        learned_kernel = create_kernel_np(kernel_network)
        if args.hard_k == True:
            learned_kernel /= kerl1(kernel_network).detach().cpu().numpy()
    else:
        learned_kernel = kernel_network.module.generatekernel().detach().cpu().numpy()[0,0]

    c = 1/10
    fig, ax = plt.subplots(1, 2, figsize = [10, 4])
    im=ax[0].imshow(learned_kernel)
    fig.colorbar(im, ax=ax[0],fraction=0.046, pad=0.04)
    im=ax[1].imshow(kernel)
    fig.colorbar(im, ax=ax[1],fraction=0.046, pad=0.04)    
    ax[0].set_title("Learned Kernel")
    ax[1].set_title("True Kernel")
    plt.savefig("{}/LearnedKernelInit.png".format(args.PATH))
    plt.close()
    
    img_blur_ext = torch.cat(args.btsize*[img_blur.unsqueeze(0)])

    for k in range(args.num_epochs):
        print("Kernel Norm", torch.sum(kernel_network.module.generatekernel()))
        print("Unnormalized Kernel Norm", torch.sum(kernel_network.module.generatekernel(nohardk=True)))
        ########## E STEP Update Generator Network ##############
        for k_sub in range(args.num_subepochsE):
            z_sample = generate_sample(args.model_name, args.btsize, npix, args.device, z_shapes)
#             z_sample = torch.randn(args.btsize, npix*npix).to(device=args.device)
            
            Eloss, qloss, priorloss, mseloss = EStep(z_sample, args.device, img_blur_ext, img_generator, kernel_network, prior, 
                                                     logdet_weight, imgtv_weight, args.data_sigma, npix, 
                                                     logscale_factor, data_weight, args.hard_k, 
                                                     kerl1)
#             print(qloss, priorloss, mseloss)
            Eloss_list.append(Eloss.detach().cpu().numpy())
            Eloss_prior_list.append(priorloss.detach().cpu().numpy())
            Eloss_q_list.append(qloss.detach().cpu().numpy())
            Eloss_mse_list.append(mseloss.detach().cpu().numpy())
            Eoptimizer.zero_grad()
            Eloss.backward()
            nn.utils.clip_grad_norm_(list(img_generator.parameters())+ list(logscale_factor.parameters()), 1)
            Eoptimizer.step()

            if ((k_sub%args.save_every == 0) and args.EMFull) or ((k%args.save_every == 0) and  not args.EMFull):
                
                n_sample = 128#1024#
                z_sample = generate_sample(args.model_name, args.btsize, npix, args.device, z_shapes)
#                 z_sample = torch.randn(n_sample, npix*npix).to(device=args.device)
                img, logdet = GForward(z_sample, img_generator, npix, logscale_factor)
                image = img.detach().cpu().numpy()

                image_stitch = np.zeros((npix*2, npix*2))
                for idx in range(4):
                    i = idx//2
                    j = idx%2
                    image_stitch[npix*i:npix*(i+1), npix*j:npix*(j+1)] = image[idx]
                plt.figure(figsize = (10,10)), plt.imshow(image_stitch, vmin=0, vmax=1), plt.title('DPI generated true image samples')
                plt.savefig("{}/ReconSamplesX{}_{}.png".format(args.PATH,str(k).zfill(5),str(k_sub).zfill(5)))
                plt.close()
                
                
                
                z_sample = generate_sample(args.model_name, args.btsize, npix, args.device, z_shapes)
                img, logdet = GForward(z_sample, img_generator, npix, logscale_factor)
                image = img.detach().cpu().numpy()

                mean_img = np.mean(image, axis=0)
                stdev_img = np.std(image, axis=0)
                stdev_img_max = stdev_img.max()
                fig, ax = plt.subplots(1, 6, figsize=(16, 4))
                im0=ax[0].imshow(img_true.detach().cpu().numpy(), vmin=0, vmax=1)
                im1=ax[1].imshow(img_blur.detach().cpu().numpy(), vmin=0, vmax=1)
                im2=ax[2].imshow(mean_img, vmin = 0, vmax = 1)
                im3=ax[3].imshow(stdev_img, vmin = 0, vmax = stdev_img_max)
                im4=ax[4].imshow(np.abs(mean_img - img_true.detach().cpu().numpy()), vmin = 0, vmax = 3*stdev_img_max)
                im5=ax[5].imshow(np.abs(mean_img - img_true.detach().cpu().numpy())/stdev_img, vmin = 0, vmax = 5)

                fig.colorbar(im0, ax=ax[0],fraction=0.046, pad=0.04)
                fig.colorbar(im1, ax=ax[1],fraction=0.046, pad=0.04)
                fig.colorbar(im2, ax=ax[2],fraction=0.046, pad=0.04)
                fig.colorbar(im3, ax=ax[3],fraction=0.046, pad=0.04)
                fig.colorbar(im4, ax=ax[4],fraction=0.046, pad=0.04)
                fig.colorbar(im5, ax=ax[5],fraction=0.046, pad=0.04)

                ax[0].set_title("true image TV-{:.2f}".format(prior(img_true.reshape([1, npix, npix]), 1).item()))
                ax[1].set_title("blurry image")
                ax[2].set_title("recon mean TV-{:.2f}".format(prior(Tensor(mean_img).reshape([1, npix, npix]), 1).item()))
                ax[3].set_title("recon stdev")
                ax[4].set_title("mean recon error")
                ax[5].set_title("mean recon error/ stdev")
                for i in range(6):
                    ax[i].axis("off")
                plt.tight_layout()
                plt.savefig("{}/ImageDist{}_{}.png".format(args.PATH,str(k).zfill(5),str(k_sub).zfill(5)))
                
                
                if k_sub != 0 and k%10 == 0:
                    with torch.no_grad():
                        torch.save({
                            'epoch':k,
                            'model_state_dict': img_generator.module.state_dict(),
                            'optimizer_state_dict': Eoptimizer.state_dict(),
                            }, '{}/{}{}_{}.pt'.format(args.PATH,"GeneratorNetwork",str(k).zfill(5),str(k_sub).zfill(5)))

            if ((k_sub%args.print_every == 0) and args.EMFull) or ((k%args.print_every == 0) and not args.EMFull):
                print(f"epoch: {k:} {k_sub:}, E step loss: {Eloss_list[-1]:.5f}")

        ########## M STEP Update Kernel Network ##############
        
        for k_sub in range(args.num_subepochsM):
            if args.model_name == 'realnvp':
                z_sample = generate_sample(args.model_name, args.btsize, npix, args.device, z_shapes)
#             z_sample = torch.randn(args.btsize, npix*npix).to(device=args.device)
            if args.x_rand == True:
#                 x_sample = torch.randn(args.btsize, npix, npix).to(device=args.device)
                x_sample = generate_sample(args.model_name, args.btsize, npix, args.device, z_shapes)
            else:
                x_sample = Tensor(sample_x_deadleaf(16, npix)).to(device=args.device)
            Mloss, philoss, mse, kernorm, priorphi = MStep(z_sample, x_sample, npix, args.device, img_blur_ext, 
                                                           img_generator, kernel_network,args.phi_weight, 
                                                           FTrue, args.data_sigma, args.kernel_norm_weight,logscale_factor,
                                                             ker_softl1, kerl1, args.hard_k, f_phi_prior, args.prior_phi_weight)

            Mloss_list.append(Mloss.detach().cpu().numpy())
            Mloss_mse_list.append(mse.detach().cpu().numpy())
            Mloss_phi_list.append(philoss)
            Mloss_phiprior_list.append(priorphi)
            Mloss_kernorm_list.append(kernorm)
            Moptimizer.zero_grad()
            Mloss.backward()
            nn.utils.clip_grad_norm_(list(kernel_network.parameters()), 1)
            Moptimizer.step()

            if ((k_sub%args.save_every == 0) and args.EMFull) or ((k%args.save_every == 0) and not args.EMFull):
               
                if args.softplus == False:
                    learned_kernel = create_kernel_np(kernel_network)
                    if args.hard_k == True:
                        learned_kernel /= kerl1(kernel_network).detach().cpu().numpy()
                else:
                    learned_kernel = kernel_network.module.generatekernel().detach().cpu().numpy()[0,0]
                    
                
                
                c = 1/10
                fig, ax = plt.subplots(1, 2, figsize = [10, 4])
                im=ax[1].imshow(kernel)
                fig.colorbar(im, ax=ax[1],fraction=0.046, pad=0.04)
                im=ax[0].imshow(learned_kernel)
                fig.colorbar(im, ax=ax[0],fraction=0.046, pad=0.04)
                ax[0].set_title("Learned Kernel")
                ax[1].set_title("True Kernel")
                plt.savefig("{}/LearnedKernel{}_{}.png".format(args.PATH, str(k).zfill(5), str(k_sub).zfill(5)))
                plt.close()

                with torch.no_grad():
                    torch.save({
                        'epoch':k,
                        'model_state_dict': kernel_network.module.state_dict(),
                        'optimizer_state_dict': Moptimizer.state_dict(),
                        }, '{}/{}{}_{}.pt'.format(args.PATH,"KernelNetwork",str(k).zfill(5),str(k_sub).zfill(5)))
                np.save("{}/Data/learned_kernel.npy".format(args.PATH), learned_kernel)
                
                fig, ax = plt.subplots()        
                plt.plot(np.log10(Eloss_list), label="Estep All")
                plt.plot(np.log10(Eloss_prior_list), ":", label="p(x)")
                plt.plot(np.log10(Eloss_mse_list), "--", label= "E step mse")
                plt.plot(np.log10(Eloss_q_list), ":", label='q')

                plt.plot(np.log10(Mloss_list), label="Mstep")
                plt.plot(np.log10(Mloss_phi_list), ":", label="p(phi)")
                plt.plot(np.log10(Mloss_mse_list), ":", label="Mstep MSE")
                plt.plot(np.log10(Mloss_kernorm_list), ":", label="Mstep Kernel Norm")
                plt.plot(np.log10(Mloss_phiprior_list), ":", label="Mstep Lp Prior")
                plt.legend()

                plt.savefig("{}/loss.png".format(args.PATH))
                plt.close()

                fig, ax = plt.subplots(1, 2, figsize = (15, 4))
                ax[0].plot(np.log10(Eloss_list), label="Estep")
                ax[0].plot(np.log10(Eloss_prior_list), ":", label="p(x)")
                ax[0].plot(np.log10(Eloss_mse_list), "--", label = "Estep mse")
                ax[0].plot(np.log10(Eloss_q_list), ":", label = "q")
                ax[0].legend()

                ax[1].plot(np.log10(Mloss_list), label="Mstep")
                ax[1].plot(np.log10(Mloss_phi_list), ":", label="p(phi)")
                ax[1].plot(np.log10(Mloss_mse_list), ":", label="Mstep MSE")
                ax[1].plot(np.log10(Mloss_kernorm_list), ":", label="Mstep Kernel Norm")
                ax[1].plot(np.log10(Mloss_phiprior_list), ":", label="Mstep Lp Prior")
                # plt.plot(loss_conv_list)
                ax[1].legend()
                plt.savefig("{}/SeparatedLoss.png".format(args.PATH))
                plt.close()
            
            if ((k_sub%args.print_every == 0) and args.EMFull) or ((k%args.print_every == 0) and not  args.EMFull):
                print(f"epoch: {k:} {k_sub:}, M step loss: {Mloss_list[-1]:.5f}")
                

    fig, ax = plt.subplots()        
    plt.plot(np.log10(Eloss_list), label="Estep All")
    plt.plot(np.log10(Eloss_prior_list), ":", label="p(x)")
    plt.plot(np.log10(Eloss_mse_list), "--", label= "E step mse")
    plt.plot(np.log10(Eloss_q_list), ":", label='q')

    plt.plot(np.log10(Mloss_list), label="Mstep")
    plt.plot(np.log10(Mloss_phi_list), ":", label="p(phi)")
    plt.plot(np.log10(Mloss_mse_list), ":", label="Mstep MSE")
    plt.plot(np.log10(Mloss_kernorm_list), ":", label="Mstep Kernel Norm")
    plt.plot(np.log10(Mloss_phiprior_list), ":", label="Mstep Lp Prior")
    plt.legend()
    
    plt.savefig("{}/loss.png".format(args.PATH))
    plt.close()
    
    fig, ax = plt.subplots(1, 2, figsize = (15, 4))
    ax[0].plot(np.log10(Eloss_list), label="Estep")
    ax[0].plot(np.log10(Eloss_prior_list), ":", label="p(x)")
    ax[0].plot(np.log10(Eloss_mse_list), "--", label = "Estep mse")
    ax[0].plot(np.log10(Eloss_q_list), ":", label = "q")
    ax[0].legend()

    ax[1].plot(np.log10(Mloss_list), label="Mstep")
    ax[1].plot(np.log10(Mloss_phi_list), ":", label="p(phi)")
    ax[1].plot(np.log10(Mloss_mse_list), ":", label="Mstep MSE")
    ax[1].plot(np.log10(Mloss_kernorm_list), ":", label="Mstep Kernel Norm")
    ax[1].plot(np.log10(Mloss_phiprior_list), ":", label="Mstep Lp Prior")
    # plt.plot(loss_conv_list)
    ax[1].legend()
    plt.savefig("{}/SeparatedLoss.png".format(args.PATH))
    plt.close()
    ############################################# GENERATE FIGURES ###########################################################

    n_sample = max(int(512*(32*32)/(args.npix**2)), 16)#1024#
#     learned_kernel = create_kernel_np(kernel_network)
#     if args.hard_k == True:
#         learned_kernel /= kerl1(kernel_network).detach().cpu().numpy()
        
    if args.softplus == False:
        learned_kernel = create_kernel_np(kernel_network)
        if args.hard_k == True:
            learned_kernel /= kerl1(kernel_network).detach().cpu().numpy()
    else:
        learned_kernel = kernel_network.module.generatekernel().detach().cpu().numpy()[0,0]
  
    c = 1/3
    fig, ax = plt.subplots(1, 2, figsize = [10, 4])
    im=ax[1].imshow(kernel, vmin = -c, vmax = c)
    fig.colorbar(im, ax=ax[1],fraction=0.046, pad=0.04)
    im=ax[0].imshow(learned_kernel, vmin=-c, vmax=c)
    fig.colorbar(im, ax=ax[0],fraction=0.046, pad=0.04)
    ax[0].set_title("Learned Kernel")
    ax[1].set_title("True Kernel")
    plt.savefig("{}/LearnedKernel.png".format(args.PATH))
    print("kernel norm", np.sum(learned_kernel))

    z_sample = generate_sample(args.model_name, args.btsize, npix, args.device, z_shapes)
#     z_sample = torch.randn(n_sample, npix*npix).to(device=args.device)
    img, logdet = GForward(z_sample, img_generator, npix, logscale_factor)
    image = img.detach().cpu().numpy()
    np.save("{}/Data/reconX.npy".format(args.PATH),image)
    print(np.sum(image)/n_sample, torch.sum(img_true))

    mean_img = np.mean(image, axis=0)
    stdev_img = np.std(image, axis=0)
    stdev_img_max = stdev_img.max()
    fig, ax = plt.subplots(1, 6, figsize=(16, 4))
    im0=ax[0].imshow(img_true.detach().cpu().numpy(), vmin=0, vmax=1)
    im1=ax[1].imshow(img_blur.detach().cpu().numpy(), vmin=0, vmax=1)
    im2=ax[2].imshow(mean_img, vmin = 0, vmax = 1)
    im3=ax[3].imshow(stdev_img, vmin = 0, vmax = stdev_img_max)
    im4=ax[4].imshow(np.abs(mean_img - img_true.detach().cpu().numpy()), vmin = 0, vmax = 3*stdev_img_max)
    im5=ax[5].imshow(np.abs(mean_img - img_true.detach().cpu().numpy())/stdev_img, vmin = 0, vmax = 5)

    fig.colorbar(im0, ax=ax[0],fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=ax[1],fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax[2],fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax[3],fraction=0.046, pad=0.04)
    fig.colorbar(im4, ax=ax[4],fraction=0.046, pad=0.04)
    fig.colorbar(im5, ax=ax[5],fraction=0.046, pad=0.04)

    ax[0].set_title("true image TV-{:.2f}".format(prior(img_true.reshape([1, npix, npix]), 1).item()))
    ax[1].set_title("blurry image")
    ax[2].set_title("recon mean TV-{:.2f}".format(prior(Tensor(mean_img).reshape([1, npix, npix]), 1).item()))
    ax[3].set_title("recon stdev")
    ax[4].set_title("mean recon error")
    ax[5].set_title("mean recon error/ stdev")
    for i in range(6):
        ax[i].axis("off")
    plt.tight_layout()
    plt.savefig("{}/MeanStdevTrue.png".format(args.PATH))

#     mean_img = np.mean(image, axis=0)
#     mean_img_norm = mean_img/np.sum(mean_img)
#     stdev_img_norm = np.std(image, axis=0)/np.sum(mean_img)
#     stdev_img_max = stdev_img_norm.max()
#     img_true_norm = img_true.detach().cpu().numpy()/np.sum(img_true.detach().cpu().numpy())
#     fig, ax = plt.subplots(1, 5, figsize=(14, 4))
#     im0=ax[0].imshow(img_true_norm, vmin=0, vmax=0.003)
#     im1=ax[1].imshow(mean_img_norm, vmin = 0, vmax = 0.003)
#     im2=ax[2].imshow(stdev_img_norm, vmin = 0, vmax = stdev_img_max)
#     im3=ax[3].imshow(np.abs(mean_img_norm - img_true_norm), vmin = 0, vmax = 3*stdev_img_max)
#     im4=ax[4].imshow(np.abs(mean_img_norm - img_true_norm)/stdev_img_norm, vmin = 0, vmax = 5)

#     fig.colorbar(im0, ax=ax[0],fraction=0.046, pad=0.04)
#     fig.colorbar(im1, ax=ax[1],fraction=0.046, pad=0.04)
#     fig.colorbar(im2, ax=ax[2],fraction=0.046, pad=0.04)
#     fig.colorbar(im3, ax=ax[3],fraction=0.046, pad=0.04)
#     fig.colorbar(im4, ax=ax[4],fraction=0.046, pad=0.04)

#     ax[0].set_title("true image")
#     ax[1].set_title("recon mean")
#     ax[2].set_title("recon stdev")
#     ax[3].set_title("mean recon error")
#     ax[4].set_title("mean recon error/ stdev")
#     for i in range(4):
#         ax[i].axis("off")
#     plt.tight_layout()
#     plt.savefig("{}/MeanStdevTrueNormalized.png".format(args.PATH))

    image_stitch = np.zeros((npix*2, npix*2))
    for k in range(4):
        i = k//2
        j = k%2
        image_stitch[npix*i:npix*(i+1), npix*j:npix*(j+1)] = image[k]
    plt.figure(figsize = (10,10)), plt.imshow(image_stitch), plt.title('DPI generated true image samples')
    plt.savefig("{}/ReconSamplesX.png".format(args.PATH))

    np.mean(np.abs(image_stitch[0:npix, 0:npix] - img_true.cpu().numpy())**2)

    y = FForward(img, kernel_network, args.data_sigma, args.device, args.hard_k, kerl1)
    image_blur = y.detach().cpu().numpy()
#     image_blur_mean_out = convolve2d(mean_img, learned_kernel)[6:-6, 6:-6]
#     image_blur_mean_out_network = np.squeeze(kernel_network(torch.unsqueeze(torch.unsqueeze(Tensor(mean_img), dim=0), dim=0).to(args.device))).cpu().detach().numpy()
#     image_blur_mean_out_true_network = np.squeeze(FTrue(torch.unsqueeze(torch.unsqueeze(Tensor(mean_img), dim=0), dim=0).to(args.device))).cpu().detach().numpy()
    np.save("{}/Data/blurY.npy".format(args.PATH), image_blur)       


    mean_blur_img = np.mean(image_blur, axis=0)
    stdev_blur_img = np.std(image_blur, axis=0)
    std_max = stdev_blur_img.max()
    fig, ax = plt.subplots(1, 5, figsize = (15, 4))
    im0=ax[0].imshow(img_blur.detach().cpu().numpy(), vmin=0, vmax=1)
    im1=ax[1].imshow(mean_blur_img, vmin=0, vmax=1)
    im2=ax[2].imshow(stdev_blur_img, vmin=0, vmax=std_max)
    im3=ax[3].imshow(np.abs(mean_blur_img - img_blur.detach().cpu().numpy()), vmin=0, vmax=5*std_max)
    im4=ax[4].imshow(np.abs(mean_blur_img - img_blur.detach().cpu().numpy())/stdev_blur_img, vmin=0, vmax=5)


    im = [im0, im1, im2, im3, im4]
    for i in range(5):
        fig.colorbar(im[i], ax=ax[i],fraction=0.046, pad=0.04)
        ax[i].axis("off")

    ax[0].set_title("true blur image")
    ax[1].set_title("mean image")
    ax[2].set_title("image stdev")
    ax[3].set_title("mean error")
    ax[4].set_title("mean error/stdev error")
    plt.tight_layout()

    plt.savefig("{}/MeanStdevBlurredY.png".format(args.PATH))


    image_stitch = np.zeros((npix*2, npix*4))
    for k in range(8):
        i = k//4
        j = k%4
        image_stitch[npix*i:npix*(i+1), npix*j:npix*(j+1)] = image_blur[k]
    plt.figure(figsize = (10,10)), plt.imshow(image_stitch, vmin=0, vmax=1), plt.title('DPI generated measured image samples')
    plt.savefig("{}/LearnedSamplesY.png".format(args.PATH))

    if args.EMFull == True:
        generate_gif(args.PATH, "LearnedKernel", args.num_epochs, args.num_subepochsM, 1, args.save_every)
        generate_gif(args.PATH, "ReconSamplesX", args.num_epochs, args.num_subepochsE, 1, args.save_every)
    else:
        generate_gif(args.PATH, "LearnedKernel", args.num_epochs, args.num_subepochsM,  args.save_every, 1)
        generate_gif(args.PATH, "ReconSamplesX", args.num_epochs, args.num_subepochsE,  args.save_every, 1)

if __name__ == "__main__":
      
    parser = argparse.ArgumentParser(description='EikoNet with Residual')
    parser.add_argument('--btsize', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 752)')
    parser.add_argument('--num_epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: 3500)')
    parser.add_argument('--num_subepochsE', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 400)')
    parser.add_argument('--num_subepochsM', type=int, default=600, metavar='N',
                        help='number of epochs to train (default: 600)')
    parser.add_argument('--save_every', type=int, default=250, metavar='N',
                        help='checkpoint model (default: 100)')
    parser.add_argument('--print_every', type=int, default=50, metavar='N',
                        help='checkpoint model (default: 50)')
    parser.add_argument('--npix', type=int, default=32, metavar='N',
                        help='size of image (default: 32)')
    parser.add_argument('--num_layers', type=int, default=3, metavar='N',
                        help='number of layers for kernel (default: 3)')
    parser.add_argument('--layer_size', type=int, default=7, metavar='N',
                        help='number of layers for kernel (default: 7)')
    
    parser.add_argument('--dv', type=str, default='cuda:1',
                        help='enables CUDA training')
    parser.add_argument('--multidv', type=int, nargs='+', default = None,
                       help="use multiple gpus (default: 1) use -1 for all")
    parser.add_argument('-dir', '--dir', type=str, default="Test",
                        help='output folder')
    parser.add_argument('--mdir', type=str, default=None,
                        help='output folder')
    
    # Training method configurations
#     parser.add_argument('--DPI', action='store_true', default=False, 
#                         help='use DPI (true forward)')
    parser.add_argument('--image', type=str, default="DeadLeaf",
                        help='image file')
    parser.add_argument('--kernel', type=str, default=None,
                        help='kernel file')
    parser.add_argument('--EMFull', action='store_true', default=False, 
                        help='True: E to convergence, M to convergence False: alternate E, M every epoch (default: False)')
    
    # network setup
    parser.add_argument('--load', action='store_true', default=False,
                        help='output folder')
    parser.add_argument('--x_rand', action='store_true', default=False,
                        help='random x or from a certain sample')
    parser.add_argument('--model_name', type=str, default="realnvp",
                        help='model type (Default: realnvp)')
    parser.add_argument('--reverse', action='store_true', default=False,
                        help='model type (Default: realnvp)')

    
    # parameters
    parser.add_argument('--tv', type=float, default=1e3, 
                        help='weight on TV')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--Elr', type=float, default=1e-4, 
                        help='learning rate(default: 1e-4)')
    parser.add_argument('--Mlr', type=float, default=1e-6, 
                        help='learning rate(default: 1e-4)')
    
    parser.add_argument('--px_weight', type=float, default=1e3, 
                        help='weight on TV')
    parser.add_argument('--kernel_norm_weight', type=float, default=0, 
                        help='fix kernel norm weight (default: 1e-5)')
    parser.add_argument('--data_sigma', type=float, default=1e-2, 
                    help='data sigma (default: 1e-2)')
    parser.add_argument('--phi_weight', type=float, default=1e-3, 
                    help='weight on prior function (default: 1e-3)')
    parser.add_argument('--hard_k', action='store_true', default=False, 
                    help='hard constraint on L1 = 1')
    parser.add_argument('--softplus', action='store_true', default=False, 
                    help='softplus output of NN')
    parser.add_argument('--softplusbeta', type=float, default=1, 
                    help='softplus output of NN')
    parser.add_argument('--padding_mode', type=str, default="reflect", 
                    help='image padding for blurry image (default:reflect)')
    parser.add_argument('--seqfracpow', type=int, default=3, 
                    help='seqfracpow (default:3)')
    parser.add_argument('--n_block', type=int, default=4, 
                    help='number of convolution sizes (default:4)')
    parser.add_argument('--prior_phi_weight', type=float, default=1e-3, 
                    help='weight on prior function (default: 1e-3)')
    parser.add_argument('--prior_phi', type=float, default=0.8, 
                    help='use lp prior (default: 0.8)')
    args = parser.parse_args()    
    
    if args.mdir is not None:
        args.PATH = "/scratch/agao3/DeconvDPIResults/EM/{}/{}_{}".format(args.mdir, args.image, args.dir)
    else:
        args.PATH = "/scratch/agao3/DeconvDPIResults/EM/{}_{}".format(args.image, args.dir)
        
    if torch.cuda.is_available():
        args.device = args.dv
        dv = int(args.device[-1])
        if args.multidv == -1:
            arr = [i for i in range(torch.cuda.device_count())]
            args.device_ids = [dv] + arr[0:dv] + arr[dv+1:]
        elif args.multidv is None:
            args.device_ids = [dv]
        else:
            args.device_ids = [dv] + args.multidv
    else:
        args.device = 'cpu'        
    
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
    print("Using {} GPUS".format(len(args.device_ids))
    
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



