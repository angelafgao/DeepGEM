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
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler,WeightedRandomSampler

from scipy.signal import convolve2d
from generative_model import realnvpfc_model
import argparse
from seismo_helpers import *


           
######################################################################################################################


class EikoNet(torch.nn.Module):
    
"""
    EikoNet and possible sine activation
    Based off of EikoNet: Solving the Eikonal equation with Deep Neural Networks
    by Jonathan D. Smith, Kamyar Azizzadenesheli, Zachary E. Ross
    
"""

    def __init__(self, input_size = 2, sine_activation = False, sine_freq = 1):
    """ Initializing the model layers with 4 resnet layers
    
    Arguments: 
        input_size: dimension of space (2 for 2D, 3 for 3D)
        sine_activation: use sine as activation layer, else use ELU
        sine_freq: frequency used for sine activation

    """
        super(EikoNet, self).__init__()
        self.sine_activation = sine_activation
        if sine_activation == True:
            print("Using Sine Activation")
            self.act = lambda x: torch.sin(sine_freq*x)
        else:
            self.act = torch.nn.ELU()
        
        # Layers 
        self.fc0  = Linear(2*input_size,32)
        self.fc1  = Linear(32,512)

        # resnet - block 1
        self.rn1_fc1  = Linear(512,512)
        self.rn1_fc2  = Linear(512,512)
        self.rn1_fc3  = Linear(512,512)

        # resnet - block 2 
        self.rn2_fc1  = Linear(512,512)
        self.rn2_fc2  = Linear(512,512)
        self.rn2_fc3  = Linear(512,512)

        # resnet - block 2 
        self.rn3_fc1  = Linear(512,512)
        self.rn3_fc2  = Linear(512,512)
        self.rn3_fc3  = Linear(512,512)

        # resnet - block 2 
        self.rn4_fc1  = Linear(512,512)
        self.rn4_fc2  = Linear(512,512)
        self.rn4_fc3  = Linear(512,512)

        # Output structure
        self.fc8  = Linear(512,32)
        self.fc9  = Linear(32,1)
        self.fc10 = Linear(1, 1)
    
    def load(self,filepath, device):
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)

    def generate_tau(self, Xsrc, Xrec, nopairs = False):  
    """ Outputs travel time multiplier (tau) learned from network 
    
    travel times = tau*T0
    
    Arguments: 
        Xsrc: source locations of size     btsz x nsrc x d
        Xrec: receiver locations of size   btsz x nrec x d
        nopairs: output travel times between same number of sources and receivers
        
    Outputs:
        tau: travel time multiplier 
        T0: distance between sources and receivers
        s: shape of Xsrc: btsz x nsrc x nrec x d
        Xsrc: resulting sources
        Xrec: resulting receivers
        
    """
        
        nsrc = Xsrc.shape[1]
        nrec = Xrec.shape[1]
        
        if nopairs == True:
            Xrec = torch.unsqueeze(Xrec, axis=2)
            Xsrc = torch.unsqueeze(Xsrc, axis=2)
        else:
            Xrec = torch.cat(nsrc*[torch.unsqueeze(Xrec, axis=1)], dim=1)
            Xsrc = torch.cat(nrec*[torch.unsqueeze(Xsrc, axis=2)], dim=2)
        
        # Xsrc shape = btsz x nsrc x nrec x d
        # Xrec shape = btsz x nsrc x nrec x d
                
        T0 = torch.sqrt(torch.sum((Xrec- Xsrc)**2+1e-8, dim = -1))
        s = Xsrc.shape
        src = Xsrc.reshape([s[0]*s[1]*s[2], s[3]])
        rec = Xrec.reshape([s[0]*s[1]*s[2], s[3]])
        
        x    = torch.cat([src, rec], dim=1)
        
        x   = self.act(self.fc0(x))
        x   = self.act(self.fc1(x))

        # Resnet - Block 1
        x0  = x
        x   = self.act(self.rn1_fc1(x))
        x   = self.act(self.rn1_fc3(x) + self.rn1_fc2(x0))

        # Resnet - Block 2
        x0  = x
        x   = self.act(self.rn2_fc1(x))
        x   = self.act(self.rn2_fc3(x)+self.rn2_fc2(x0))

        # Resnet - Block 3
        x0  = x
        x   = self.act(self.rn3_fc1(x))
        x   = self.act(self.rn3_fc3(x)+self.rn3_fc2(x0))

        # Resnet - Block 4
        x0  = x
        x   = self.act(self.rn4_fc1(x))
        x   = self.act(self.rn4_fc3(x)+self.rn4_fc2(x0))

        # Joining two blocks
        x     = self.act(self.fc8(x))
        tau   = torch.abs(self.fc10(self.fc9(x)))
        return tau, T0, s, Xsrc, Xrec
    
    def generate_velo(self, Xsrc, Xrec, device, retain_graph, vinvar):
    """ Outputs velocity learned from network
    
    Arguments: 
        Xsrc: source locations of size     btsz x nsrc x d
        Xrec: receiver locations of size   btsz x nrec x d
        device: device of model
        retain_graph: true for L_T, false otherwise
        vinvar: used for computing the priors L_V, L_theta
        
    Outputs:
        Vmat: matrix of velocity
        tau*T0: travel times

    """
        tau, t0, s, XsrcFull, XrecFull = self.generate_tau(Xsrc, Xrec)
        tau = torch.squeeze(tau,axis=1)
        T0 = t0.reshape([s[1]*s[2]])
        
        dtau_mat  = torch.squeeze(torch.autograd.grad(outputs=tau, inputs=XrecFull, grad_outputs=torch.ones(tau.size()).to(device), 
                        only_inputs=True,create_graph=True,retain_graph=retain_graph)[0], axis=0)
        dtau = dtau_mat.reshape([s[1]*s[2], s[3]])
        XsrcExt = XsrcFull.reshape([s[1]*s[2], s[3]])
        XrecExt = XrecFull.reshape([s[1]*s[2], s[3]])
        
        T1    = (T0**2)*(torch.sum(dtau**2, axis=1))
        T2    = 2*tau*(dtau[:,0]*(XrecExt[:,0]-XsrcExt[:,0]) + dtau[:,1]*(XrecExt[:,1]-XsrcExt[:,1]))
        T3    = tau**2
        S2    = (T1+T2+T3)
        V = (1/S2+1e-8)**(1/2)
        
        if vinvar == True:
            Vmat = V.reshape([s[1], s[2]])
        else:
            Vmat = V.reshape([1, s[1], s[2]])
        return Vmat, tau*T0
    
    def init_model(self, fwdmodel, rand, device):#, init_prior = False):
    """
    Initialize model with pretrained networks or use random intialization

    Arguments: 
        fwdmodel: name of forward model used for pretraining
        rand: randomly initilize network
        device: device of network

    """
        if self.sine_activation == True:
#             self.apply(init_weights_eiko_sine)
#             print("Init with random model")
#             if init_prior == True:
#                 self.load(("SeismoGEMResults/EM/EMTestsGradBlur/"
#                            "EIKOSampled_FwdHs_data1e-2_xsigma1e-1_px_randmean1e-1_nsrc100_nrec20/"
#                             "ForwardNetwork10000_00000.pt"), device)
#                 print("Init with Model Trained with Uncertain Prior")
#             else:
            if fwdmodel == "GradBlur1.3":
                self.load(("SeismoGEMResults/FNet_Pretrain/"
                            "EIKO_VeloLoss_GradBlur1.3_prior1e-2_data1e-3_nsrc100_nrec100/"
                            "ForwardNetwork30000_00000.pt"), device)         
                print("Init with GradBlur1.3 model")
            elif fwdmodel == "GradBlur1.3Grad" or fwdmodel == "GRFGrad":
                self.load(("SeismoGEMResults/FNet_Pretrain/"
                            "EIKO_VeloLoss_NewGrad_prior1e-2_data1e-3_nsrc100_nrec100/"
                            "ForwardNetwork10000_00000.pt"), device)         
                print("Init with GradBlur1.3Grad NewGrad model")  
            elif fwdmodel == "GradBlobBlur1.3_0.3":
                self.load(("SeismoGEMResults/FNet_Pretrain/"
                            "EIKO_VeloLoss_GradBlur1.3Blob3_prior1e-2_data1e-3_nsrc100_nrec100/"
                            "ForwardNetwork50000_00000.pt"), device)  

            elif fwdmodel == "H6":
                self.load(("SeismoGEMResults/FNet_Pretrain/"
                            "EIKO_VeloLoss_H6_prior1e-2_data1e-3_nsrc100_nrec100/"
                            "ForwardNetwork10000_00000.pt"), device)         
                print("Init with H6 model")
            else:
                self.load("SeismoGEMResults/FNet_Pretrain/Init_nsrc100_nrec100/ForwardNetwork00300.pt", device)
                print("Init with H5 model")
        else:
            self.apply(init_weights_eiko)
            print("init with random model")
            if rand == False:
                self.load("SeismoGEMResults/Init_nsrc100_nrec100/ForwardNetwork00300.pt", device)
                print("Init with forward model")
        return
   
    def forward(self, Xsrc, Xrec, device, v = None, nopairs = False, velo=False, retain_graph=True, vinvar=False): 
    """
    
    Arguments: 
        Xsrc: source locations   btsz x nsrc x d
        Xrec: receiver locations btsz x nrec x d
        device: device of network
        v: assumed velocity for homogeneous model
        nopairs: output travel times between same number of sources and receivers
        velo: true to output velocity of rec locations relative to source, false to compute travel times between sources and receivers
        retain_graph: true for L_T, false otherwise
        vinvar: used for computing the priors L_V, L_theta

    """
        if velo == False:
            tau, T0, s, _, _ = self.generate_tau(Xsrc, Xrec, nopairs)

            tau = tau.reshape([s[0], s[1], s[2]])
            TT = tau*T0
            return TT
        else:
            V, TT = self.generate_velo(Xsrc, Xrec, device, retain_graph, vinvar=vinvar)
            return V
