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
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
# import torchvision.utils as vutils
from torchsummary import summary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

import argparse
import time
from datetime import datetime
import glob
import pickle
import yaml

# Import modules from other files
from utils import *
from spec_loss import *

def f_parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run script to train GAN using LBANN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument
    
    add_arg('--epochs','-e', type=int, default=10, help='The number of epochs')
    add_arg('--ngpu','-ngpu',  type=int, default=1, help='The number of GPUs per node to use')
    add_arg('--mode','-m',  type=str, choices=['fresh','continue'],default='fresh', help='Whether to start fresh run or continue previous run')
    add_arg('--ip_fldr','-ip',  type=str, default='', help='The input folder for resuming a checkpointed run')
    add_arg('--run_suffix','-rs',  type=str, default='train', help='String to attach at the end of the run')
    add_arg('--config','-cfile',  type=str, default='/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/cosmogan_pytorch/cosmogan/main_code/config_128.yaml', help='Whether to start fresh run or continue previous run')
    add_arg('--seed','-s',  type=str, default='random', help='Seed for random number sequence')
    add_arg('--batchsize','-b',  type=int, default=64, help='batchsize')
    add_arg('--learn_rate','-lr',  type=float, default=0.0002, help='Learn rate')
    add_arg('--lambda1','-ld1',  type=float, default=1.0, help='Coupling for spectral loss values')
    add_arg('--specloss','-spl',action='store_true', help='Whether to use spectral loss')
    return parser.parse_args()

def f_train_loop(dataloader,metrics_df,start_epoch,num_epochs,iters,best_chi1,best_chi2,save_dir):
    ''' Train single epoch '''
    
    for epoch in range(start_epoch,num_epochs):
        t_epoch_start=time.time()
        for count, data in enumerate(dataloader, 0):
            
            ####### Train GAN ########
            netG.train(); netD.train();  ### Need to add these after inference and before training

            tme1=time.time()
            ### Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()

            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            real_label = torch.full((b_size,), 1, device=device)
            fake_label = torch.full((b_size,), 0, device=device)
            g_label = torch.full((b_size,), 1, device=device) ## No flipping for Generator labels
            # Flip labels with probability flip_prob
            for idx in np.random.choice(np.arange(b_size),size=int(np.ceil(b_size*flip_prob))):
                real_label[idx]=0; fake_label[idx]=1

            # Generate fake image batch with G
            noise = torch.randn(b_size, 1, 1, nz, device=device)
            fake = netG(noise)            

            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, real_label)
            errD_real.backward()
            D_x = output.mean().item()

            # Forward pass real batch through D
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, fake_label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ###Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            output = netD(fake).view(-1)
            errG_adv = criterion(output, g_label)
            # Histogram pixel intensity loss
            hist_gen=f_compute_hist(fake,bins=bns)
            hist_loss=loss_hist(hist_gen,hist_val.to(device))

            # Add spectral loss
            mean,sdev=f_torch_image_spectrum(f_invtransform(fake),1,r.to(device),ind.to(device))
            spec_loss=loss_spectrum(mean,mean_spec_val.to(device),sdev,sdev_spec_val.to(device),image_size)
            
            if spec_loss_flag: errG=errG_adv+spec_loss
            else: errG=errG_adv

            if torch.isnan(errG).any():
                print(errG)
                raise SystemError
            
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            tme2=time.time()

            ####### Store metrics ########
            # Output training stats
            if count % checkpoint_size == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_adv: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, count, len(dataloader), errD.item(), errG_adv.item(),errG.item(), D_x, D_G_z1, D_G_z2)),
                print("Spec loss: %s,\t hist loss: %s"%(spec_loss.item(),hist_loss.item())),
                print("Training time for step %s : %s"%(iters, tme2-tme1))

            # Save metrics
            cols=['step','epoch','Dreal','Dfake','Dfull','G_adv','G_full','spec_loss','hist_loss','D(x)','D_G_z1','D_G_z2','time']
            vals=[iters,epoch,errD_real.item(),errD_fake.item(),errD.item(),errG_adv.item(),errG.item(),spec_loss.item(),hist_loss.item(),D_x,D_G_z1,D_G_z2,tme2-tme1]
            for col,val in zip(cols,vals):  metrics_df.loc[iters,col]=val

            ### Checkpoint the best model
            checkpoint=True
            iters += 1  ### Model has been updated, so update iters before saving metrics and model.

            ### Compute validation metrics for updated model
            netG.eval()
            with torch.no_grad():
                #fake = netG(fixed_noise).detach().cpu()
                fake = netG(fixed_noise)
                hist_gen=f_compute_hist(fake,bins=bns)
                hist_chi=loss_hist(hist_gen,hist_val.to(device))
                mean,sdev=f_torch_image_spectrum(f_invtransform(fake),1,r.to(device),ind.to(device))
                spec_chi=loss_spectrum(mean,mean_spec_val.to(device),sdev,sdev_spec_val.to(device),image_size)      
            # Storing chi for next step
            for col,val in zip(['spec_chi','hist_chi'],[spec_chi.item(),hist_chi.item()]):  metrics_df.loc[iters,col]=val            

            # Checkpoint model for continuing run
            if count == len(dataloader)-1: ## Check point at last step of epoch
                f_save_checkpoint(epoch,iters,best_chi1,best_chi2,netG,netD,optimizerG,optimizerD,save_loc=save_dir+'/models/checkpoint_last.tar')  

            if (checkpoint and (epoch > 1)): # Choose best models by metric
                if hist_chi< best_chi1:
                    f_save_checkpoint(epoch,iters,best_chi1,best_chi2,netG,netD,optimizerG,optimizerD,save_loc=save_dir+'/models/checkpoint_best_hist.tar')
                    best_chi1=hist_chi.item()
                    print("Saving best hist model at epoch %s, step %s."%(epoch,iters))

                if  spec_chi< best_chi2:
                    f_save_checkpoint(epoch,iters,best_chi1,best_chi2,netG,netD,optimizerG,optimizerD,save_loc=save_dir+'/models/checkpoint_best_spec.tar')
                    best_chi2=spec_chi.item()
                    print("Saving best spec model at epoch %s, step %s"%(epoch,iters))

            # Save G's output on fixed_noise
            if ((iters % checkpoint_size == 0) or ((epoch == num_epochs-1) and (count == len(dataloader)-1))):
                netG.eval()
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    img_arr=np.array(fake[:,0,:,:])
                    fname='gen_img_epoch-%s_step-%s'%(epoch,iters)
                    np.save(save_dir+'/images/'+fname,img_arr)
        
        t_epoch_end=time.time()
        print("Time taken for epoch %s: %s"%(epoch,t_epoch_end-t_epoch_start))
        # Save Metrics to file after each epoch
        metrics_df.to_pickle(save_dir+'/df_metrics.pkle')
        
    print("best chis",best_chi1,best_chi2)

#########################
### Main code #######
#########################

if __name__=='__main__':
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic=True
    torch.autograd.set_detect_anomaly(True)
    
    t0=time.time()
    ###############
    ### Set up ###
    args=f_parse_args()
    print('Args',args)
    ngpu,num_epochs=args.ngpu,args.epochs
    mode,batch_size=args.mode,args.batchsize
    config_file=args.config
    spec_loss_flag=args.specloss

    config_dict=f_load_config(config_file)
    print(config_dict)
    
    # Initilize variables
    workers=config_dict['training']['workers']
    nc,nz,ngf,ndf=config_dict['training']['nc'],config_dict['training']['nz'],config_dict['training']['ngf'],config_dict['training']['ndf']
    beta1=config_dict['training']['beta1']
    kernel_size,stride=config_dict['training']['kernel_size'],config_dict['training']['stride']
    g_padding,d_padding=config_dict['training']['g_padding'],config_dict['training']['d_padding']
    flip_prob=config_dict['training']['flip_prob']

    image_size=config_dict['data']['image_size']
    checkpoint_size=config_dict['data']['checkpoint_size']
    num_imgs=config_dict['data']['num_imgs']
    ip_fname=config_dict['data']['ip_fname']
    op_loc=config_dict['data']['op_loc']
    
    if spec_loss_flag: print("Using Spectral loss")
    lr=args.learn_rate
    bns=50
    img_size=2000
    
    ### Initialize random seed
    if args.seed=='random': manualSeed = np.random.randint(1, 10000)
    else: manualSeed=int(args.seed)
    print("Seed: ", manualSeed)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print('Device:',device)
 
    #################################
    ####### Read data and precompute ######
    # ip_fname='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/128_square/dataset_2_smoothing_200k/norm_1_train_val.npy'
    img=np.load(ip_fname)[:num_imgs].transpose(0,1,2,3)
    t_img=torch.from_numpy(img)
    print(img.shape,t_img.shape)

    dataset=TensorDataset(t_img)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=1,drop_last=True)

    # Precompute metrics with validation data for computing losses
    with torch.no_grad():
        val_img=np.load(ip_fname)[-3000:].transpose(0,1,2,3)
        t_val_img=torch.from_numpy(val_img)

        # Precompute radial coordinates
        r,ind=f_get_rad(img)
        # Stored mean and std of spectrum for full input data once
        mean_spec_val,sdev_spec_val=f_torch_image_spectrum(f_invtransform(t_val_img),1,r,ind)
        hist_val=f_compute_hist(t_val_img,bins=bns)
        del val_img; del t_val_img; del img; del t_img

    #################################
    ###### Build Networks ###
    print("Building GAN networks")
    # Create Generator
    netG = Generator(ngpu,nz,nc,ngf,kernel_size,stride,g_padding).to(device)
    netG.apply(weights_init)
    print(netG)
    summary(netG,(1,1,64))
    # Create Discriminator
    netD = Discriminator(ngpu, nz,nc,ndf,kernel_size,stride,g_padding).to(device)
    netD.apply(weights_init)
    print(netD)
    summary(netD,(1,128,128))
    # Handle multi-gpu if desired
    ngpu=torch.cuda.device_count()

    print("Number of GPUs used",ngpu)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Initialize BCELoss function
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    fixed_noise = torch.randn(batch_size, 1, 1, nz, device=device) #Latent vectors to view G progress
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999),eps=1e-7)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999),eps=1e-7)
    
    #################################    
    ###### Set up directories #######
    ## For checkpointed runs, 
    if args.mode=='fresh':
        # Create prefix for foldername 
        now=datetime.now()
        fldr_name=now.strftime('%Y%m%d_%H%M%S') ## time format
        # print(fldr_name)
        save_dir=op_loc+fldr_name+'_'+args.run_suffix
        if not os.path.exists(save_dir):
            os.makedirs(save_dir+'/models')
            os.makedirs(save_dir+'/images')
            
        ### Initialize variables
        iters = 0; start_epoch=0
        best_chi1,best_chi2=1e10,1e10
        
    elif args.mode=='continue':
        save_dir=args.ip_fldr
        iters,start_epoch,best_chi1,best_chi2=f_load_checkpoint(save_dir+'/models/checkpoint_last.tar',netG,netD,optimizerG,optimizerD) 
        print("Continuing existing run. Loading checkpoint with epoch {0} and step {1}".format(start_epoch,iters))
        start_epoch+=1  ## Start with the next epoch   
        
        ### Read loss data
        with open (save_dir+'/metrics.pickle','rb') as f:
            metrics_dict=pickle.load(f) 
    
    ### Set up metrics dataframe
    cols=['step','epoch','Dreal','Dfake','Dfull','G_adv','G_full','spec_loss','hist_loss','spec_chi','hist_chi','D(x)','D_G_z1','D_G_z2','time']
    # size=int(len(dataloader) * num_epochs)+1
    metrics_df=pd.DataFrame(columns=cols)

    #################################
    ########## Train loop and save metrics and images ######
    print("Starting Training Loop...")
    f_train_loop(dataloader,metrics_df,start_epoch,num_epochs,iters,best_chi1,best_chi2,save_dir)

    ## Generate images for best saved models ######
    model_fname=save_dir+'/models/checkpoint_best_spec.tar'
    f_gen_images(netG,optimizerG,nz,device,model_fname,'spec',save_dir,2000)

    model_fname=save_dir+'/models/checkpoint_best_hist.tar'
    f_gen_images(netG,optimizerG,nz,device,model_fname,'hist',save_dir,2000)  

    tf=time.time()
    print("Total time",tf-t0)
    