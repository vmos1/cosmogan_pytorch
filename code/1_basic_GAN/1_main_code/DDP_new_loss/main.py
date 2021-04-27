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
#from torchsummary import summary
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
    args.config='config_2dgan.yaml'
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
    
    add_arg('--config','-cfile',  type=str, default='config_2dgan.yaml', help='Whether to start fresh run or continue previous run')
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

def f_sample_data(ip_tensor,rank=0,num_ranks=1):
    '''
    Module to load part of dataset depending on world_rank.
    '''
    data_size=ip_tensor.shape[0]
    size=data_size//num_ranks
    logging.info("Using data indices %s-%s for rank %s"%(rank*(size),(rank+1)*size,rank))
    dataset=TensorDataset(ip_tensor[rank*(size):(rank+1)*size])
    ### 
    if gdict['batch_size']>size:
        print("Caution: batchsize %s is less than samples per GPU."%(gdict['batch_size'],size))
        raise SystemExit
    
    data_loader=DataLoader(dataset,batch_size=gdict['batch_size'],shuffle=True,num_workers=0,drop_last=True)
    return data_loader


def f_load_data_precompute(gdict):
    #################################
    ####### Read data and precompute ######
    img=np.load(gdict['ip_fname'],mmap_mode='r')[:gdict['num_imgs']].transpose(0,1,2,3).copy()
    t_img=torch.from_numpy(img)
#     print("%s, %s"%(img.shape,t_img.shape))

#     dataset=TensorDataset(t_img)
#     data_loader=DataLoader(dataset,batch_size=gdict['batch_size'],shuffle=True,num_workers=0,drop_last=True)

    data_loader=f_sample_data(t_img,gdict['world_rank'],gdict['world_size'])
    logging.info("Size of dataset for GPU %s : %s"%(gdict['world_rank'],len(data_loader.dataset)))
    
    # Precompute metrics with validation data for computing losses
    with torch.no_grad():
        val_img=np.load(gdict['ip_fname'],mmap_mode='r')[-3000:].transpose(0,1,2,3).copy()
        t_val_img=torch.from_numpy(val_img).to(gdict['device'])

        # Precompute radial coordinates
        r,ind=f_get_rad(img)
        r=r.to(gdict['device']); ind=ind.to(gdict['device'])
        # Stored mean and std of spectrum for full input data once
        mean_spec_val,sdev_spec_val=f_torch_image_spectrum(f_invtransform(t_val_img),1,r,ind)
        hist_val=f_compute_hist(t_val_img,bins=gdict['bns'])
        del val_img; del t_val_img; del img; del t_img

    return data_loader,mean_spec_val,sdev_spec_val,hist_val,r,ind

def f_init_GAN(gdict,print_model=False):
    # Define Models
    logging.info("Building GAN networks")
    # Create Generator
    netG = Generator(gdict).to(gdict['device'])
    netG.apply(weights_init)
    # Create Discriminator
    netD = Discriminator(gdict).to(gdict['device'])
    netD.apply(weights_init)
    
    if print_model:
        if gdict['world_rank']==0:
            print(netG)
        #     summary(netG,(1,1,64))
            print(netD)
        #     summary(netD,(1,128,128))
            print("Number of GPUs used %s"%(gdict['ngpu']))

    if (gdict['multi-gpu']):
        if not gdict['distributed']:
            netG = nn.DataParallel(netG, list(range(gdict['ngpu'])))
            netD = nn.DataParallel(netD, list(range(gdict['ngpu'])))
        else:
            netG=DistributedDataParallel(netG,device_ids=[gdict['local_rank']],output_device=[gdict['local_rank']])
            netD=DistributedDataParallel(netD,device_ids=[gdict['local_rank']],output_device=[gdict['local_rank']])
    
    #### Initialize networks ####
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    if gdict['mode']=='fresh':
        optimizerD = optim.Adam(netD.parameters(), lr=gdict['learn_rate'], betas=(gdict['beta1'], 0.999),eps=1e-7)
        optimizerG = optim.Adam(netG.parameters(), lr=gdict['learn_rate'], betas=(gdict['beta1'], 0.999),eps=1e-7)
        ### Initialize variables      
        iters,start_epoch,best_chi1,best_chi2=0,0,1e10,1e10    

    ### Load network weights for continuing run
    elif gdict['mode']=='continue':
        iters,start_epoch,best_chi1,best_chi2=f_load_checkpoint(gdict['save_dir']+'/models/checkpoint_last.tar',netG,netD,optimizerG,optimizerD,gdict) 
        logging.info("Continuing existing run. Loading checkpoint with epoch {0} and step {1}".format(start_epoch,iters))
        start_epoch+=1  ## Start with the next epoch  

    ## Add to gdict
    for key,val in zip(['best_chi1','best_chi2','iters','start_epoch'],[best_chi1,best_chi2,iters,start_epoch]): gdict[key]=val

    return netG,netD,criterion,optimizerD,optimizerG

def f_setup(gdict,log):
    ''' 
    Set up directories, Initialize random seeds, add GPU info, add logging info.
    '''
    
    torch.backends.cudnn.benchmark=True
    torch.autograd.set_detect_anomaly(True)

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

### Train code ###

def f_train_loop(dataloader,metrics_df,gdict,fixed_noise,mean_spec_val,sdev_spec_val,hist_val,r,ind):
    ''' Train epochs '''
#     print("Inside train loop")

    ## Define new variables from dict
    keys=['image_size','start_epoch','epochs','iters','best_chi1','best_chi2','save_dir','device','flip_prob','nz','batch_size','bns']
    image_size,start_epoch,epochs,iters,best_chi1,best_chi2,save_dir,device,flip_prob,nz,batchsize,bns=list(collections.OrderedDict({key:gdict[key] for key in keys}).values())
    
    for epoch in range(start_epoch,epochs):
        t_epoch_start=time.time()
        for count, data in enumerate(dataloader):

            ####### Train GAN ########
            netG.train(); netD.train();  ### Need to add these after inference and before training

            tme1=time.time()
            ### Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()

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
            noise = torch.randn(b_size, 1, 1, nz, device=device)
            fake = netG(noise)            

            # Forward pass real batch through D
            real_output = netD(real_cpu)
            errD_real = criterion(real_output[-1].view(-1), real_label.float())
            errD_real.backward(retain_graph=True)
            D_x = real_output[-1].mean().item()

            # Forward pass fake batch through D
            fake_output = netD(fake.detach())   # The detach is important
            errD_fake = criterion(fake_output[-1].view(-1), fake_label.float())
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

            if gdict['grad_clip']: ### Implement Gradient clipping
                nn.utils.clip_grad_norm_(netD.parameters(),gdict['grad_clip'])
            optimizerD.step()

            ###Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            output = netD(fake)
            errG_adv = criterion(output[-1].view(-1), g_label.float())
#             errG_adv.backward(retain_graph=True)
            # Histogram pixel intensity loss
            hist_gen=f_compute_hist(fake,bins=bns)
            hist_loss=loss_hist(hist_gen,hist_val.to(device))

            # Add spectral loss
            mean,sdev=f_torch_image_spectrum(f_invtransform(fake),1,r.to(device),ind.to(device))
            spec_loss=loss_spectrum(mean,mean_spec_val.to(device),sdev,sdev_spec_val.to(device),image_size,gdict['lambda_spec_mean'],gdict['lambda_spec_var'])

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
                nn.utils.clip_grad_norm_(netG.parameters(),gdict['grad_clip'])           
            optimizerG.step()

            tme2=time.time()
            ####### Store metrics ########
            # Output training stats
            if gdict['world_rank']==0:
                if ((count % gdict['checkpoint_size'] == 0)):
                    logging.info('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_adv: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, epochs, count, len(dataloader), errD.item(), errG_adv.item(),errG.item(), D_x, D_G_z1, D_G_z2)),
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
                netG.eval()
                with torch.no_grad():
                    fake = netG(fixed_noise)
                    hist_gen=f_compute_hist(fake,bins=bns)
                    hist_chi=loss_hist(hist_gen,hist_val.to(device))
                    mean,sdev=f_torch_image_spectrum(f_invtransform(fake),1,r.to(device),ind.to(device))
                    spec_chi=loss_spectrum(mean,mean_spec_val.to(device),sdev,sdev_spec_val.to(device),image_size,gdict['lambda_spec_mean'],gdict['lambda_spec_var'])      
                # Storing chi for next step
                for col,val in zip(['spec_chi','hist_chi'],[spec_chi.item(),hist_chi.item()]):  metrics_df.loc[iters,col]=val            

                # Checkpoint model for continuing run
                if count == len(dataloader)-1: ## Check point at last step of epoch
                    f_save_checkpoint(gdict,epoch,iters,best_chi1,best_chi2,netG,netD,optimizerG,optimizerD,save_loc=save_dir+'/models/checkpoint_last.tar')  

                if (checkpoint and (epoch > 1)): # Choose best models by metric
                    if hist_chi< best_chi1:
                        f_save_checkpoint(gdict,epoch,iters,best_chi1,best_chi2,netG,netD,optimizerG,optimizerD,save_loc=save_dir+'/models/checkpoint_best_hist.tar')
                        best_chi1=hist_chi.item()
                        logging.info("Saving best hist model at epoch %s, step %s."%(epoch,iters))

                    if  spec_chi< best_chi2:
                        f_save_checkpoint(gdict,epoch,iters,best_chi1,best_chi2,netG,netD,optimizerG,optimizerD,save_loc=save_dir+'/models/checkpoint_best_spec.tar')
                        best_chi2=spec_chi.item()
                        logging.info("Saving best spec model at epoch %s, step %s"%(epoch,iters))

                    if iters in gdict['save_steps_list']:
                        f_save_checkpoint(gdict,epoch,iters,best_chi1,best_chi2,netG,netD,optimizerG,optimizerD,save_loc=save_dir+'/models/checkpoint_{0}.tar'.format(iters))
                        logging.info("Saving given-step at epoch %s, step %s."%(epoch,iters))

                # Save G's output on fixed_noise
                if ((iters % gdict['checkpoint_size'] == 0) or ((epoch == epochs-1) and (count == len(dataloader)-1))):
                    netG.eval()
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                        img_arr=np.array(fake)
                        fname='gen_img_epoch-%s_step-%s'%(epoch,iters)
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
#     gdict['num_imgs']=5000

    if jpt: ## override for jpt nbks
        gdict['num_imgs']=400
        gdict['run_suffix']='nb_test'
        
    f_setup(gdict,log=(not jpt))
    
    ## Build GAN
    netG,netD,criterion,optimizerD,optimizerG=f_init_GAN(gdict,print_model=True)
#     fixed_noise = torch.randn(gdict['batch_size']*gdict['world_size'], 1, 1, gdict['nz'], device=gdict['device']) #Latent vectors to view G progress 
    fixed_noise = torch.randn(64, 1, 1, gdict['nz'], device=gdict['device']) #Latent vectors to view G progress 

    if gdict['distributed']:  try_barrier(gdict['world_rank'])

    ## Load data and precompute
    dataloader,mean_spec_val,sdev_spec_val,hist_val,r,ind=f_load_data_precompute(gdict)

    #################################
    ########## Train loop and save metrics and images ######
    ### Set up metrics dataframe
    cols=['step','epoch','Dreal','Dfake','Dfull','G_adv','G_full','spec_loss','hist_loss','spec_chi','hist_chi','gp_loss','fm_loss','D(x)','D_G_z1','D_G_z2','time']
    metrics_df=pd.DataFrame(columns=cols)
    if gdict['distributed']:  try_barrier(gdict['world_rank'])

#     print(gdict)
    logging.info("Starting Training Loop...")
    f_train_loop(dataloader,metrics_df,gdict,fixed_noise,mean_spec_val,sdev_spec_val,hist_val,r,ind)
    
    if gdict['world_rank']==0: ## Generate images for best saved models ######
        op_loc=gdict['save_dir']+'/images/'
        ip_fname=gdict['save_dir']+'/models/checkpoint_best_spec.tar'
        f_gen_images(gdict,netG,optimizerG,ip_fname,op_loc,op_strg='best_spec',op_size=200)
        ip_fname=gdict['save_dir']+'/models/checkpoint_best_hist.tar'
        f_gen_images(gdict,netG,optimizerG,ip_fname,op_loc,op_strg='best_hist',op_size=200)
    
    tf=time.time()
    logging.info("Total time %s"%(tf-t0))
    logging.info('End: %s'%(datetime.now().strftime('%Y-%m-%d  %H:%M:%S')))



