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

# March 30, 2021 
# Adding pytorch distributed data parallel and new loss functions 

import os
import random
import logging
import sys
import subprocess

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
# from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
# import torch.fft

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from IPython.display import HTML

import argparse
import time
from datetime import datetime
import glob
import pickle
import yaml
import collections
import socket
import shutil

# Import modules from other files
from utils import *
from spec_loss import *


### Setup modules ###
def f_manual_add_argparse():
    ''' use only in jpt notebook'''
    args=argparse.Namespace()
    args.config='config_3dgan.yaml'
    args.mode='fresh'
    args.ip_fldr=''
#     args.local_rank=0
    args.facility='cori'
    args.distributed=False
#     args.mode='continue'
#     args.ip_fldr='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/results_from_other_code/pytorch/results/128sq/20201211_093818_nb_test/'
    
    return args

def f_parse_args():
    """Parse command line arguments.Only for .py file"""
    parser = argparse.ArgumentParser(description="Run script to train GAN using pytorch", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument
    
    add_arg('--config','-cfile',  type=str, default='config_3dgan.yaml', help='Whether to start fresh run or continue previous run')
    add_arg('--mode','-m',  type=str, choices=['fresh','continue'],default='fresh', help='Whether to start fresh run or continue previous run')
    add_arg('--ip_fldr','-ip',  type=str, default='', help='The input folder for resuming a checkpointed run')
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

            
def f_setup(gdict,log):
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
        
        logging.info("World size %s, world rank %s, local rank %s, hostname %s, GPUs on node %s\n"%(gdict['world_size'],gdict['world_rank'],gdict['local_rank'],socket.gethostname(),gdict['ngpu']))
        device = torch.cuda.current_device()
        
        # Divide batch size by number of GPUs
#         gdict['batch_size']=gdict['batch_size']//gdict['world_size']
    else:
        gdict['world_size'],gdict['world_rank'],gdict['local_rank']=1,0,0
    
    ########################
    ###### Set up directories #######
    ### sync up so that time is the same for each GPU for DDP
    if gdict['mode']=='fresh':
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
        gdict['save_dir']=args.ip_fldr
        ### Read loss data
        with open (gdict['save_dir']+'df_metrics.pkle','rb') as f:
            metrics_dict=pickle.load(f)
   
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

class Dataset:
    def __init__(self,gdict):
        '''
        Load training dataset and compute spectrum and histogram for a small batch of training and validation dataset.
        '''
        ## Load training dataset
        t0a=time.time()
        img=np.load(gdict['ip_fname'],mmap_mode='r')[:gdict['num_imgs']]
    #     print("Shape of input file",img.shape)
        img=f_get_img_samples(img,gdict['world_rank'],gdict['world_size'])  

        t_img=torch.from_numpy(img)
        dataset=TensorDataset(t_img)
        self.train_dataloader=DataLoader(dataset,batch_size=gdict['batch_size'],shuffle=True,num_workers=0,drop_last=True)
        logging.info("Size of dataset for GPU %s : %s"%(gdict['world_rank'],len(self.train_dataloader.dataset)))

        t0b=time.time()
        logging.info("Time for creating dataloader %s for rank %s"%(t0b-t0a,gdict['world_rank']))
        
        # Precompute spectrum and histogram for small training and validation data for computing losses
        with torch.no_grad():
            val_img=np.load(gdict['ip_fname'],mmap_mode='r')[-100:].copy()
            t_val_img=torch.from_numpy(val_img).to(gdict['device'])
            # Precompute radial coordinates
            r,ind=f_get_rad(val_img)
            self.r,self.ind=r.to(gdict['device']),ind.to(gdict['device'])

            # Compute
            self.train_spec_mean,self.train_spec_var=f_torch_image_spectrum(f_invtransform(t_val_img),1,self.r,self.ind)
            self.train_hist=f_compute_hist(t_val_img,bins=gdict['bns'])
            
            # Repeat for validation dataset
            val_img=np.load(gdict['ip_fname'],mmap_mode='r')[-200:-100].copy()
            t_val_img=torch.from_numpy(val_img).to(gdict['device'])
            
            # Compute
            self.val_spec_mean,self.val_spec_var=f_torch_image_spectrum(f_invtransform(t_val_img),1,self.r,self.ind)
            self.val_hist=f_compute_hist(t_val_img,bins=gdict['bns'])
            del val_img; del t_val_img; del img; del t_img;


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

        if gdict['mode']=='fresh':
            self.optimizerD = optim.Adam(self.netD.parameters(), lr=gdict['learn_rate'], betas=(gdict['beta1'], 0.999),eps=1e-7)
            self.optimizerG = optim.Adam(self.netG.parameters(), lr=gdict['learn_rate'], betas=(gdict['beta1'], 0.999),eps=1e-7)
            ### Initialize variables      
            iters,start_epoch,best_chi1,best_chi2=0,0,1e10,1e10    

        ### Load network weights for continuing run
        elif gdict['mode']=='continue':
            iters,start_epoch,best_chi1,best_chi2=f_load_checkpoint(gdict['save_dir']+'/models/checkpoint_last.tar',self.netG,self.netD,self.optimizerG,self.optimizerD,gdict) 
            logging.info("Continuing existing run. Loading checkpoint with epoch {0} and step {1}".format(start_epoch,iters))
            start_epoch+=1  ## Start with the next epoch 

        ## Add to gdict
        for key,val in zip(['best_chi1','best_chi2','iters','start_epoch'],[best_chi1,best_chi2,iters,start_epoch]): gdict[key]=val


### Train code ###

def f_train_loop(gan_model,Dset,metrics_df,gdict,fixed_noise):
    ''' Train epochs '''
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
            fake = gan_model.netG(noise)            

            # Forward pass real batch through D
            real_output = gan_model.netD(real_cpu)
            errD_real = gan_model.criterion(real_output[-1].view(-1), real_label.float())
            errD_real.backward(retain_graph=True)
            D_x = real_output[-1].mean().item()

            # Forward pass fake batch through D
            fake_output = gan_model.netD(fake.detach())   # The detach is important
            errD_fake = gan_model.criterion(fake_output[-1].view(-1), fake_label.float())
            errD_fake.backward(retain_graph=True)
            D_G_z1 = fake_output[-1].mean().item()
            
            errD = errD_real + errD_fake 

            if gdict['lambda_gp']: ## Add gradient - penalty loss
                grads=torch.autograd.grad(outputs=real_output[-1],inputs=real_cpu,grad_outputs=torch.ones_like(real_output[-1]),allow_unused=False,create_graph=True)[0]
                gp_loss=f_gp_loss(grads,gdict['lambda_gp'])
                gp_loss.backward(retain_graph=True)
                errD = errD + gp_loss
            else:
                gp_loss=torch.Tensor([np.nan])
                
            if gdict['grad_clip']:
                nn.utils.clip_grad_norm_(gan_model.netD.parameters(),gdict['grad_clip'])

            gan_model.optimizerD.step()
# dict_keys(['train_data_loader', 'r', 'ind', 'train_spec_mean', 'train_spec_var', 'train_hist', 'val_spec_mean', 'val_spec_var', 'val_hist'])

            ###Update G network: maximize log(D(G(z)))
            gan_model.netG.zero_grad()
            output = gan_model.netD(fake)
            errG_adv = gan_model.criterion(output[-1].view(-1), g_label.float())
#             errG_adv.backward(retain_graph=True)
            # Histogram pixel intensity loss
            hist_gen=f_compute_hist(fake,bins=bns)
            hist_loss=loss_hist(hist_gen,Dset.train_hist.to(device))

            # Add spectral loss
            mean,var=f_torch_image_spectrum(f_invtransform(fake),1,Dset.r.to(device),Dset.ind.to(device))
            spec_loss=loss_spectrum(mean,Dset.train_spec_mean.to(device),var,Dset.train_spec_var.to(device),image_size,gdict['lambda_spec_mean'],gdict['lambda_spec_var'])

            errG=errG_adv
            if gdict['lambda_spec_mean']: 
#                 spec_loss.backward(retain_graph=True)
                errG = errG+ spec_loss 
            if gdict['lambda_fm']:## Add feature matching loss
                fm_loss=f_FM_loss([i.detach() for i in real_output],output,gdict['lambda_fm'],gdict)
#                 fm_loss.backward(retain_graph=True)
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
                cols=['step','epoch','Dreal','Dfake','Dfull','G_adv','G_full','spec_loss','hist_loss','fm_loss','gp_loss','D(x)','D_G_z1','D_G_z2','time']
                vals=[iters,epoch,errD_real.item(),errD_fake.item(),errD.item(),errG_adv.item(),errG.item(),spec_loss.item(),hist_loss.item(),fm_loss.item(),gp_loss.item(),D_x,D_G_z1,D_G_z2,tme2-tme1]
                for col,val in zip(cols,vals):  metrics_df.loc[iters,col]=val

                ### Checkpoint the best model
                checkpoint=True
                iters += 1  ### Model has been updated, so update iters before saving metrics and model.

                ### Compute validation metrics for updated model
                gan_model.netG.eval()
                with torch.no_grad():
                    fake = gan_model.netG(fixed_noise)
                    hist_gen=f_compute_hist(fake,bins=bns)
                    hist_chi=loss_hist(hist_gen,Dset.val_hist.to(device))
                    mean,var=f_torch_image_spectrum(f_invtransform(fake),1,Dset.r.to(device),Dset.ind.to(device))
                    spec_chi=loss_spectrum(mean,Dset.val_spec_mean.to(device),var,Dset.val_spec_var.to(device),image_size,gdict['lambda_spec_mean'],gdict['lambda_spec_var'])

                # Storing chi for next step
                for col,val in zip(['spec_chi','hist_chi'],[spec_chi.item(),hist_chi.item()]):  metrics_df.loc[iters,col]=val            

                # Checkpoint model for continuing run
                if count == len(Dset.train_dataloader)-1: ## Check point at last step of epoch
                    f_save_checkpoint(gdict,epoch,iters,best_chi1,best_chi2,gan_model.netG,gan_model.netD,gan_model.optimizerG,gan_model.optimizerD,save_loc=save_dir+'/models/checkpoint_last.tar')  

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
                        fake = gan_model.netG(fixed_noise).detach().cpu()
                        img_arr=np.array(fake)
                        fname='gen_img_epoch-%s_step-%s'%(epoch,iters)
                        np.save(save_dir+'/images/'+fname,img_arr)
                
                if ((count % gdict['checkpoint_size'] == 0)):
                    tme3=time.time()
                    logging.info("Post training time for step %s : %s"%(iters, tme3-tme2))
        
        
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
        gdict['num_imgs']=400
        gdict['run_suffix']='nb_test'
        
    f_setup(gdict,log=(not jpt))
    
    ## Build GAN
    gan_model=GAN_model(gdict,False)
    fixed_noise = torch.randn(gdict['op_size'], 1, 1, 1, gdict['nz'], device=gdict['device']) #Latent vectors to view G progress    # Mod for 3D

    if gdict['distributed']:  try_barrier(gdict['world_rank'])

    ## Load data and precompute
    Dset=Dataset(gdict)
    
    #################################
    ########## Train loop and save metrics and images ######
    ### Set up metrics dataframe
    cols=['step','epoch','Dreal','Dfake','Dfull','G_adv','G_full','spec_loss','hist_loss','spec_chi','hist_chi','gp_loss','fm_loss','D(x)','D_G_z1','D_G_z2','time']
    metrics_df=pd.DataFrame(columns=cols)
    if gdict['distributed']:  try_barrier(gdict['world_rank'])

#     print(gdict)
    logging.info("Starting Training Loop...")
    f_train_loop(gan_model,Dset,metrics_df,gdict,fixed_noise)
    
    if gdict['world_rank']==0: ## Generate images for best saved models ######
        op_loc=gdict['save_dir']+'/images/'
        ip_fname=gdict['save_dir']+'/models/checkpoint_best_spec.tar'
        f_gen_images(gdict,gan_model.netG,gan_model.optimizerG,ip_fname,op_loc,op_strg='best_spec',op_size=gdict['op_size'])
        ip_fname=gdict['save_dir']+'/models/checkpoint_best_hist.tar'
        f_gen_images(gdict,gan_model.netG,gan_model.optimizerG,ip_fname,op_loc,op_strg='best_hist',op_size=gdict['op_size'])
    
    tf=time.time()
    logging.info("Total time %s"%(tf-t0))
    logging.info('End: %s'%(datetime.now().strftime('%Y-%m-%d  %H:%M:%S')))









