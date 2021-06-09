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

import argparse
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
# import torch.fft

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


def try_barrier(rank):
    """
    Used in Distributed data parallel
    Attempt a barrier but ignore any exceptions
    """
    print('BAR %d'%rank)
    try:
        dist.barrier()
    except:
        pass

def f_init_gdict(args,gdict):
    ''' Create global dictionary gdict from args and config file'''
    
    ## read config file
    config_file=args.config
    with open(config_file) as f:
        config_dict= yaml.load(f, Loader=yaml.SafeLoader)
        
    gdict=config_dict['parameters']

    args_dict=vars(args)
    ## Add args variables to gdict
    for key in args_dict.keys():
        gdict[key]=args_dict[key]

    if gdict['distributed']: 
        assert not gdict['lambda_gp'],"GP couplings is %s. Cannot use Gradient penalty loss in pytorch DDP"%(gdict['lambda_gp'])
    else : print("Not using DDP")
    return gdict


def f_get_img_samples(ip_arr,rank=0,num_ranks=1):
    '''
    Module to get part of the numpy image file
    '''
    
    data_size=ip_arr.shape[0]
    size=data_size//num_ranks
    
    if gdict['batch_size']>size:
        print("Caution: batchsize %s is greater than samples per GPU %s"%(gdict['batch_size'],size))
        raise SystemExit
        
    ### Get a set of random indices from numpy array
    random=False
    if random:
        idxs=np.arange(ip_arr.shape[0])
        np.random.shuffle(idxs)
        rnd_idxs=idxs[rank*(size):(rank+1)*size]
        arr=ip_arr[rnd_idxs].copy()
        
    else: arr=ip_arr[rank*(size):(rank+1)*size].copy()
    
    return arr

def f_setup(gdict,metrics_df,log):
    ''' 
    Set up directories, Initialize random seeds, add GPU info, add logging info.
    '''
    
    torch.backends.cudnn.benchmark=True
#     torch.autograd.set_detect_anomaly(True)

    ## New additions. Code taken from Jan B.
    os.environ['MASTER_PORT'] = "8885"

    if gdict['facility']=='summit':
        get_master = "echo $(cat {} | sort | uniq | grep -v batch | grep -v login | head -1)".format(os.environ['LSB_DJOB_HOSTFILE'])
        os.environ['MASTER_ADDR'] = str(subprocess.check_output(get_master, shell=True))[2:-3]
        os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
        os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
        gdict['local_rank'] = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    else:
        if gdict['distributed']:
            os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
            os.environ['RANK'] = os.environ['SLURM_PROCID']
            gdict['local_rank'] = int(os.environ['SLURM_LOCALID'])

    ## Special declarations
    gdict['ngpu']=torch.cuda.device_count()
    gdict['device']=torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    gdict['multi-gpu']=True if (gdict['device'].type == 'cuda') and (gdict['ngpu'] > 1) else False 
    
    ########################
    ###### Set up Distributed Data parallel ######
    if gdict['distributed']:
#         gdict['local_rank']=args.local_rank  ## This is needed when using pytorch -m torch.distributed.launch
        gdict['world_size']=int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(gdict['local_rank']) ## Very important
        dist.init_process_group(backend='nccl', init_method="env://")  
        gdict['world_rank']= dist.get_rank()
        
        device = torch.cuda.current_device()
        logging.info("World size %s, world rank %s, local rank %s device %s, hostname %s, GPUs on node %s\n"%(gdict['world_size'],gdict['world_rank'],gdict['local_rank'],device,socket.gethostname(),gdict['ngpu']))
        
        # Divide batch size by number of GPUs
#         gdict['batch_size']=gdict['batch_size']//gdict['world_size']
    else:
        gdict['world_size'],gdict['world_rank'],gdict['local_rank']=1,0,0

    
    ########################
    ###### Set up directories #######
    ### sync up so that time is the same for each GPU for DDP
    if gdict['mode'] in ['fresh','fresh_load']:
        ### Create prefix for foldername      
        if gdict['world_rank']==0: ### For rank=0, create directory name string and make directories
            dt_strg=datetime.now().strftime('%Y%m%d_%H%M%S') ## time format
            dt_lst=[int(i) for i in dt_strg.split('_')] # List storing day and time            
            dt_tnsr=torch.LongTensor(dt_lst).to(gdict['device'])  ## Create list to pass to other GPUs 

        else: dt_tnsr=torch.Tensor([0,0]).long().to(gdict['device'])
        ### Pass directory name to other ranks
        if gdict['distributed']: dist.broadcast(dt_tnsr, src=0)

        gdict['save_dir']=gdict['op_loc']+str(int(dt_tnsr[0]))+'_'+str(int(dt_tnsr[1]))+'_'+gdict['run_suffix']
        
        if gdict['world_rank']==0: # Create directories for rank 0
            ### Create directories
            if not os.path.exists(gdict['save_dir']):
                os.makedirs(gdict['save_dir']+'/models')
                os.makedirs(gdict['save_dir']+'/images')
                shutil.copy(gdict['config'],gdict['save_dir'])    
    
    elif gdict['mode']=='continue': ## For checkpointed runs
        gdict['save_dir']=gdict['ip_fldr']
        ### Read loss data
        metrics_df=pd.read_pickle(gdict['save_dir']+'/df_metrics.pkle').astype(np.float64)
   
    ########################
    ### Initialize random seed
    
    manualSeed = np.random.randint(1, 10000) if gdict['seed']=='random' else int(gdict['seed'])
#     print("Seed",manualSeed,gdict['world_rank'])
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    
    if gdict['deterministic']:
        logging.info("Running with deterministic sequence. Performance will be slower")
        torch.backends.cudnn.deterministic=True
#         torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False        
    
    ########################
    if log:
        ### Write all logging.info statements to stdout and log file
        logfile=gdict['save_dir']+'/log.log'
        if gdict['world_rank']==0:
            logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")

            Lg = logging.getLogger()
            Lg.setLevel(logging.DEBUG)
            lg_handler_file = logging.FileHandler(logfile)
            lg_handler_stdout = logging.StreamHandler(sys.stdout)
            Lg.addHandler(lg_handler_file)
            Lg.addHandler(lg_handler_stdout)

            logging.info('Args: {0}'.format(args))
            logging.info('Start: %s'%(datetime.now().strftime('%Y-%m-%d  %H:%M:%S')))
        
        if gdict['distributed']:  try_barrier(gdict['world_rank'])

        if gdict['world_rank']!=0:
                logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")

    return metrics_df

class Dataset:
    def __init__(self,gdict):
        '''
        Load training dataset and compute spectrum and histogram for a small batch of training and validation dataset.
        '''
        
        ## Load training dataset
        t0a=time.time()
        for count,sigma in enumerate(gdict['sigma_list']):
            fname=gdict['ip_fname']+'/norm_1_sig_%s_train_val.npy'%(sigma)
            x=np.load(fname,mmap_mode='r')[:gdict['num_imgs']].transpose(0,1,2,3,4) ## Mod for 3D
            x=f_get_img_samples(x,gdict['world_rank'],gdict['world_size'])
            size=x.shape[0]
            y=sigma*np.ones(size)

            if count==0:
                img=x[:]
                c_pars=y[:]
            else: 
                img=np.vstack([img,x]) # Store images
                c_pars=np.hstack([c_pars,y]) # Store cosmological parameters

        ### Manually shuffling numpy arrays to mix sigma values
        size=img.shape[0]
        idxs=np.random.choice(size,size=size,replace=False)
        img=img[idxs]
        c_pars=c_pars[idxs]
        ## convert to tensors
        t_img=torch.from_numpy(img)
        cosm_params=torch.Tensor(c_pars).view(size,1)

        dataset=TensorDataset(t_img,cosm_params)
        self.train_dataloader=DataLoader(dataset,batch_size=gdict['batch_size'],shuffle=True,num_workers=0,drop_last=True)
        logging.info("Size of dataset for GPU %s : %s"%(gdict['world_rank'],len(self.train_dataloader.dataset)))

        t0b=time.time()
        logging.info("Time for creating dataloader %s for rank %s"%(t0b-t0a,gdict['world_rank']))


        # Precompute spectrum and histogram for small training and validation data for computing losses           
        def f_compute_summary_stats(idx1=-50,idx2=None):
            # Compute hist and spec for given dataset
            with torch.no_grad():
                spec_mean_list=[];spec_var_list=[];hist_val_list=[]
                for count,sigma in enumerate(gdict['sigma_list']):
                    ip_fname=gdict['ip_fname']+'/norm_1_sig_%s_train_val.npy'%(sigma)
                    val_img=np.load(ip_fname,mmap_mode='r')[idx1:idx2].transpose(0,1,2,3,4).copy() ## Mod for 3D
                    t_val_img=torch.from_numpy(val_img).to(gdict['device'])

                    # Precompute radial coordinates
                    if count==0: 
                        r,ind=f_get_rad(val_img)
                        r=r.to(gdict['device']); ind=ind.to(gdict['device'])
                    # Stored mean and std of spectrum for full input data once
                    mean_spec_val,var_spec_val=f_torch_image_spectrum(f_invtransform(t_val_img),1,r,ind)
                    hist_val=f_compute_hist(t_val_img,bins=gdict['bns'])

                    spec_mean_list.append(mean_spec_val)
                    spec_var_list.append(var_spec_val)
                    hist_val_list.append(hist_val)
    #             del val_img; del t_val_img; del img; del spec_mean_list; del spec_var_list; del hist_val_list   
                return torch.stack(spec_mean_list),torch.stack(spec_var_list),torch.stack(hist_val_list),r,ind
        
        with torch.no_grad():
            self.train_spec_mean,self.train_spec_var,self.train_hist,self.r,self.ind=f_compute_summary_stats(-50,None)
            ## Compute for validation data
            self.val_spec_mean,self.val_spec_var,self.val_hist,_,_=f_compute_summary_stats(-100,-50)

class GAN_model():
    def __init__(self,gdict,print_model=False):
    
        def weights_init(m):
            '''custom weights initialization called on netG and netD '''
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        
        ## Choose model
        Generator, Discriminator=f_get_model(gdict['model'],gdict) ## Mod for cGAN

        # Create Generator
        self.netG = Generator(gdict).to(gdict['device'])
        self.netG.apply(weights_init)
        # Create Discriminator
        self.netD = Discriminator(gdict).to(gdict['device'])
        self.netD.apply(weights_init)

        if print_model:
            if gdict['world_rank']==0:
                print(self.netG)
            #     summary(netG,(1,1,64))
                print(self.netD)
            #     summary(netD,(1,128,128))
                print("Number of GPUs used %s"%(gdict['ngpu']))

        if (gdict['multi-gpu']):
            if not gdict['distributed']:
                self.netG = nn.DataParallel(self.netG, list(range(gdict['ngpu'])))
                self.netD = nn.DataParallel(self.netD, list(range(gdict['ngpu'])))
            else:
                self.netG=DistributedDataParallel(self.netG,device_ids=[gdict['local_rank']],output_device=[gdict['local_rank']])
                self.netD=DistributedDataParallel(self.netD,device_ids=[gdict['local_rank']],output_device=[gdict['local_rank']])

        #### Initialize networks ####
        # self.criterion = nn.BCELoss()
        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=gdict['learn_rate_d'], betas=(gdict['beta1'], 0.999),eps=1e-7)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=gdict['learn_rate_g'], betas=(gdict['beta1'], 0.999),eps=1e-7)
        
        if gdict['distributed']:  try_barrier(gdict['world_rank'])

        if gdict['mode']=='fresh':
            iters,start_epoch,best_chi1,best_chi2=0,0,1e10,1e10 
            
        elif gdict['mode']=='continue':
            iters,start_epoch,best_chi1,best_chi2,self.netD,self.optimizerD,self.netG,self.optimizerG=f_load_checkpoint(gdict['save_dir']+'/models/checkpoint_last.tar',\
                                                                                                                        self.netG,self.netD,self.optimizerG,self.optimizerD,gdict) 
            if gdict['world_rank']==0: logging.info("\nContinuing existing run. Loading checkpoint with epoch {0} and step {1}\n".format(start_epoch,iters))
            if gdict['distributed']:  try_barrier(gdict['world_rank'])
            start_epoch+=1  ## Start with the next epoch 
        
        elif gdict['mode']=='fresh_load':
            iters,start_epoch,best_chi1,best_chi2,self.netD,self.optimizerD,self.netG,self.optimizerG=f_load_checkpoint(gdict['chkpt_file'],\
                                                                                                                        self.netG,self.netD,self.optimizerG,self.optimizerD,gdict) 
            if gdict['world_rank']==0: logging.info("Fresh run loading checkpoint file {0}".format(gdict['chkpt_file']))
#             if gdict['distributed']:  try_barrier(gdict['world_rank'])
            iters,start_epoch,best_chi1,best_chi2=0,0,1e10,1e10 
        
        ## Add to gdict
        for key,val in zip(['best_chi1','best_chi2','iters','start_epoch'],[best_chi1,best_chi2,iters,start_epoch]): gdict[key]=val
        
        ## Set up learn rate scheduler
        lr_stepsize=int((gdict['num_imgs']*len(gdict['sigma_list']))/(gdict['batch_size']*gdict['world_size'])) # convert epoch number to step 
        lr_d_epochs=[i*lr_stepsize for i in gdict['lr_d_epochs']] 
        lr_g_epochs=[i*lr_stepsize for i in gdict['lr_g_epochs']]
        self.schedulerD = optim.lr_scheduler.MultiStepLR(self.optimizerD, milestones=lr_d_epochs,gamma=gdict['lr_d_gamma'])
        self.schedulerG = optim.lr_scheduler.MultiStepLR(self.optimizerG, milestones=lr_g_epochs,gamma=gdict['lr_g_gamma'])

### Train code ###

def f_train_loop(gan_model,Dset,metrics_df,gdict,fixed_noise,fixed_cosm_params):
    ''' Train single epoch '''

    ## Define new variables from dict
    keys=['image_size','start_epoch','epochs','iters','best_chi1','best_chi2','save_dir','device','flip_prob','nz','batch_size','bns']
    image_size,start_epoch,epochs,iters,best_chi1,best_chi2,save_dir,device,flip_prob,nz,batchsize,bns=list(collections.OrderedDict({key:gdict[key] for key in keys}).values())
    
    for epoch in range(start_epoch,epochs):
        t_epoch_start=time.time()
        for count, data in enumerate(Dset.train_dataloader):

            ####### Train GAN ########
            gan_model.netG.train(); gan_model.netD.train();  ### Need to add these after inference and before training

            tme1=time.time()
            ### Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            gan_model.netD.zero_grad()

            real_cpu = data[0].to(device)
            real_cosm_params=data[1].to(device)
            real_cpu.requires_grad=True
            
            b_size = real_cpu.size(0)
            real_label = torch.full((b_size,), 1, device=device,dtype=float)
            fake_label = torch.full((b_size,), 0, device=device,dtype=float)
            g_label = torch.full((b_size,), 1, device=device,dtype=float) ## No flipping for Generator labels
            # Flip labels with probability flip_prob
            for idx in np.random.choice(np.arange(b_size),size=int(np.ceil(b_size*flip_prob))):
                real_label[idx]=0; fake_label[idx]=1

            # Generate fake image batch with G
            noise = torch.randn(b_size, 1, 1, 1, nz, device=device) ### Mod for 3D
            rnd_idx=torch.randint(len(gdict['sigma_list']),(gdict['batch_size'],1),device=gdict['device'])
            fake_cosm_params=torch.tensor([gdict['sigma_list'][i] for i in rnd_idx.long()],device=gdict['device']).unsqueeze(-1)

            fake = gan_model.netG(noise,fake_cosm_params)         

            # Forward pass real batch through D
            real_output = gan_model.netD(real_cpu,real_cosm_params)
            errD_real = gan_model.criterion(real_output[-1].view(-1), real_label.float())
            errD_real.backward(retain_graph=True)
            D_x = real_output[-1].mean().item()

            # Forward pass fake batch through D
            fake_output = gan_model.netD(fake.detach(),fake_cosm_params)  # The detach is important
            errD_fake = gan_model.criterion(fake_output[-1].view(-1), fake_label.float())
            errD_fake.backward(retain_graph=True)
            D_G_z1 = fake_output[-1].mean().item()
            
            errD = errD_real + errD_fake 

            if gdict['lambda_gp']: ## Add gradient - penalty loss                
                grads=torch.autograd.grad(outputs=real_output[-1],inputs=real_cpu,grad_outputs=torch.ones_like(real_output[-1]),allow_unused=False,create_graph=True)[0]
                gp_loss=f_get_loss_cond('gp',fake,fake_cosm_params,gdict,grads=grads)
                gp_loss.backward(retain_graph=True)
                errD = errD + gp_loss
            else:
                gp_loss=torch.Tensor([np.nan])

            ### Implement Gradient clipping
            if gdict['grad_clip']:
                nn.utils.clip_grad_norm_(gan_model.netD.parameters(),gdict['grad_clip'])
                
            gan_model.optimizerD.step()
            lr_d=gan_model.optimizerD.param_groups[0]['lr']
            gan_model.schedulerD.step()
            
            ###Update G network: maximize log(D(G(z)))
            gan_model.netG.zero_grad()
            output = gan_model.netD(fake,fake_cosm_params)
            errG_adv = gan_model.criterion(output[-1].view(-1), g_label.float())
            # Histogram pixel intensity loss
            hist_loss=f_get_loss_cond('hist',fake,fake_cosm_params,gdict,bins=gdict['bns'],hist_val_tnsr=Dset.train_hist)

            # Add spectral loss
            mean,var=f_torch_image_spectrum(f_invtransform(fake),1,Dset.r.to(device),Dset.ind.to(device))
            spec_loss=f_get_loss_cond('spec',fake,fake_cosm_params,gdict,spec_mean_tnsr=Dset.train_spec_mean,spec_var_tnsr=Dset.train_spec_var,r=Dset.r,ind=Dset.ind)
            
            errG=errG_adv
            if gdict['lambda_spec_mean']: errG = errG+ spec_loss 
            if gdict['lambda_fm']:## Add feature matching loss
                fm_loss=f_get_loss_cond('fm',fake,fake_cosm_params,gdict,real_output=[i.detach() for i in real_output],fake_output=output)
                errG= errG+ fm_loss
            else: 
                fm_loss=torch.Tensor([np.nan])

            if torch.isnan(errG).any():
                logging.info(errG)
                raise SystemError
            
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output[-1].mean().item()

            ### Implement Gradient clipping
            if gdict['grad_clip']:
                nn.utils.clip_grad_norm_(gan_model.netG.parameters(),gdict['grad_clip'])

            gan_model.optimizerG.step()
            lr_g=gan_model.optimizerG.param_groups[0]['lr']
            gan_model.schedulerG.step()
            
            tme2=time.time()
            ####### Store metrics ########
            # Output training stats
            if gdict['world_rank']==0:
                if ((count % gdict['checkpoint_size'] == 0)):
                    logging.info('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_adv: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, epochs, count, len(Dset.train_dataloader), errD.item(), errG_adv.item(),errG.item(), D_x, D_G_z1, D_G_z2)),
                    logging.info("Spec loss: %s,\t hist loss: %s"%(spec_loss.item(),hist_loss.item())),
                    logging.info("Training time for step %s : %s"%(iters, tme2-tme1))

                # Save metrics
                cols=['step','epoch','Dreal','Dfake','Dfull','G_adv','G_full','spec_loss','hist_loss','fm_loss','gp_loss','D(x)','D_G_z1','D_G_z2','lr_d','lr_g','time']
                vals=[iters,epoch,errD_real.item(),errD_fake.item(),errD.item(),errG_adv.item(),errG.item(),spec_loss.item(),hist_loss.item(),fm_loss.item(),gp_loss.item(),D_x,D_G_z1,D_G_z2,lr_d,lr_g,tme2-tme1]
                for col,val in zip(cols,vals):  metrics_df.loc[iters,col]=val

                ### Checkpoint the best model
                checkpoint=True
                iters += 1  ### Model has been updated, so update iters before saving metrics and model.

                ### Compute validation metrics for updated model
                gan_model.netG.eval()
                with torch.no_grad():
                    fake = gan_model.netG(fixed_noise,fixed_cosm_params)
                    hist_chi=f_get_loss_cond('hist',fake,fixed_cosm_params,gdict,bins=gdict['bns'],hist_val_tnsr=Dset.val_hist)
                    spec_chi=f_get_loss_cond('spec',fake,fixed_cosm_params,gdict,spec_mean_tnsr=Dset.val_spec_mean,spec_var_tnsr=Dset.val_spec_var,r=Dset.r,ind=Dset.ind)

                # Storing chi for next step
                for col,val in zip(['spec_chi','hist_chi'],[spec_chi.item(),hist_chi.item()]):  metrics_df.loc[iters,col]=val            

                # Checkpoint model for continuing run
                if count == len(Dset.train_dataloader)-1: ## Checkpoint at last step of epoch
                    f_save_checkpoint(gdict,epoch,iters,best_chi1,best_chi2,gan_model.netG,gan_model.netD,gan_model.optimizerG,gan_model.optimizerD,save_loc=save_dir+'/models/checkpoint_last.tar')  
                    shutil.copy(save_dir+'/models/checkpoint_last.tar',save_dir+'/models/checkpoint_%s_%s.tar'%(epoch,iters)) # Store last step for each epoch
                    
                if (checkpoint and (epoch > 1)): # Choose best models by metric
                    if hist_chi< best_chi1:
                        f_save_checkpoint(gdict,epoch,iters,best_chi1,best_chi2,gan_model.netG,gan_model.netD,gan_model.optimizerG,gan_model.optimizerD,save_loc=save_dir+'/models/checkpoint_best_hist.tar')
                        best_chi1=hist_chi.item()
                        logging.info("Saving best hist model at epoch %s, step %s."%(epoch,iters))

                    if  spec_chi< best_chi2:
                        f_save_checkpoint(gdict,epoch,iters,best_chi1,best_chi2,gan_model.netG,gan_model.netD,gan_model.optimizerG,gan_model.optimizerD,save_loc=save_dir+'/models/checkpoint_best_spec.tar')
                        best_chi2=spec_chi.item()
                        logging.info("Saving best spec model at epoch %s, step %s"%(epoch,iters))

#                    if (iters in gdict['save_steps_list']) :
                    if ((gdict['save_steps_list']=='all') and (iters % gdict['checkpoint_size'] == 0)):                        
                        f_save_checkpoint(gdict,epoch,iters,best_chi1,best_chi2,gan_model.netG,gan_model.netD,gan_model.optimizerG,gan_model.optimizerD,save_loc=save_dir+'/models/checkpoint_{0}.tar'.format(iters))
                        logging.info("Saving given-step at epoch %s, step %s."%(epoch,iters))

                # Save G's output on fixed_noise
                if ((iters % gdict['checkpoint_size'] == 0) or ((epoch == epochs-1) and (count == len(Dset.train_dataloader)-1))):
                    gan_model.netG.eval()
                    with torch.no_grad():
                        for c_pars in gdict['sigma_list']:
                            tnsr_cosm_params=(torch.ones(gdict['op_size'],device=device)*c_pars).view(gdict['op_size'],1)
                            fake = gan_model.netG(fixed_noise,tnsr_cosm_params).detach().cpu()
                            img_arr=np.array(fake)
                            fname='gen_img_label-%s_epoch-%s_step-%s'%(c_pars,epoch,iters)
                            np.save(save_dir+'/images/'+fname,img_arr)
        
        t_epoch_end=time.time()
        if gdict['world_rank']==0:
            logging.info("Time taken for epoch %s, count %s: %s for rank %s"%(epoch,count,t_epoch_end-t_epoch_start,gdict['world_rank']))
            # Save Metrics to file after each epoch
            metrics_df.to_pickle(save_dir+'/df_metrics.pkle')
            logging.info("best chis: {0}, {1}".format(best_chi1,best_chi2))


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
#     gdict['num_imgs']=200

    if jpt: ## override for jpt nbks
        gdict['num_imgs']=100
        gdict['run_suffix']='nb_test'
        
    ### Set up metrics dataframe
    cols=['step','epoch','Dreal','Dfake','Dfull','G_adv','G_full','spec_loss','hist_loss','spec_chi','hist_chi','gp_loss','fm_loss','D(x)','D_G_z1','D_G_z2','time']
    metrics_df=pd.DataFrame(columns=cols)
    
    # Setup
    metrics_df=f_setup(gdict,metrics_df,log=(not jpt))
    
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

