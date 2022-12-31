#!/usr/bin/env python
# coding: utf-8

# # Testing cosmogan
# Sep 4, 2020
# Author: Venkitesh Ayyar. vpa@lbl.gov
# 
# Borrowing pieces of code from : 
# 
# - https://github.com/pytorch/tutorials/blob/11569e0db3599ac214b03e01956c2971b02c64ce/beginner_source/dcgan_faces_tutorial.py
# - https://github.com/exalearn/epiCorvid/tree/master/cGAN

import os
import sys
import subprocess
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
# import torchvision.utils as vutils
# from torchsummary import summary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import argparse
import time
from datetime import datetime
import glob
import pickle
import yaml
import shutil
import socket

import logging
import collections

# Import modules from other files
from utils import *
from spec_loss import *
from modules_main import *

########## Modules
### Setup modules ###
def f_manual_add_argparse():
    ''' use only in jpt notebook'''
    args=argparse.Namespace()
    args.config='config_3d_Cgan.yaml'
    args.mode='fresh'
    args.local_rank=0
    args.facility='cori'
    args.distributed=False

#     args.mode='continue'
    
    return args

def f_parse_args():
    """Parse command line arguments.Only for .py file"""
    parser = argparse.ArgumentParser(description="Run script to train GAN using pytorch", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument
    
    add_arg('--config','-cfile',  type=str, default='config_3d_Cgan.yaml', help='Name of config file')
    add_arg('--mode','-m',  type=str, choices=['fresh','continue','fresh_load'],default='fresh', help='Whether to start fresh run or continue previous run or fresh run loading a config file.')
    add_arg("--local_rank", default=0, type=int,help='Local rank of GPU on node. Using for pytorch DDP. ')
    add_arg("--facility", default='cori', choices=['cori','summit'],type=str,help='Facility: cori or summit ')
    add_arg("--ddp", dest='distributed' ,default=False,action='store_true',help='use Distributed DataParallel for Pytorch or DataParallel')
    
    return parser.parse_args()

#########################
### Main code #######
#########################

if __name__=="__main__":
    jpt=False
#     jpt=True ##(different for jupyter notebook)
    t0=time.time()
    args=f_parse_args() if not jpt else f_manual_add_argparse()

    #################################
    ### Set up global dictionary###
    gdict={}
    gdict=f_init_gdict(args,gdict)
    # gdict['num_imgs']=200

    if jpt: ## override for jpt nbks
        gdict['num_imgs']=100
        gdict['run_suffix']='nb_test'
        
    ### Set up metrics dataframe
    cols=['step','epoch','Dreal','Dfake','Dfull','G_adv','G_full','spec_loss','hist_loss','spec_chi','hist_chi','gp_loss','fm_loss','D(x)','D_G_z1','D_G_z2','time']
    metrics_df=pd.DataFrame(columns=cols)
    
    # Setup
    metrics_df=f_setup(gdict,args,metrics_df,log=(not jpt))
    
    ## Build GAN
    gan_model=GAN_model(gdict,True)

    fixed_noise = torch.randn(gdict['op_size'], 1, 1, 1, gdict['nz'], device=gdict['device']) #Latent vectors to view G progress    # Mod for 3D
    rnd_idx=torch.randint(len(gdict['sigma_list']),(gdict['op_size'],1),device=gdict['device'])
    fixed_cosm_params=torch.tensor([gdict['sigma_list'][i] for i in rnd_idx.long()],device=gdict['device']).unsqueeze(-1)
    
    if gdict['distributed']:  try_barrier(gdict['world_rank'])
    
    ## Load data and precompute
    Dset=Dataset(gdict)
    
    #################################
    ########## Train loop and save metrics and images ######    
    if gdict['distributed']:  try_barrier(gdict['world_rank'])
        
    if gdict['world_rank']==0: 
        logging.info(gdict)
        logging.info("Starting Training Loop...")
    
    f_train_loop(gan_model,Dset,metrics_df,gdict,fixed_noise,fixed_cosm_params)

    if gdict['world_rank']==0: ## Generate images for best saved models ######
        for cl in gdict['sigma_list']:
            op_loc=gdict['save_dir']+'/images/'
            ip_fname=gdict['save_dir']+'/models/checkpoint_best_spec.tar'
            f_gen_images(gdict,gan_model.netG,gan_model.optimizerG,cl,ip_fname,op_loc,op_strg='gen_img_best_spec',op_size=32)

            ip_fname=gdict['save_dir']+'/models/checkpoint_best_hist.tar'
            f_gen_images(gdict,gan_model.netG,gan_model.optimizerG,cl,ip_fname,op_loc,op_strg='gen_img_best_hist',op_size=32)
    
    tf=time.time()
    logging.info("Total time %s"%(tf-t0))
    logging.info('End: %s'%(datetime.now().strftime('%Y-%m-%d  %H:%M:%S')))

