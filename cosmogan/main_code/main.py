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

import torchvision.utils as vutils
import numpy as np
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
    add_arg('--config','-cfile',  type=str, default='/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/cosmogan_pytorch/cosmogan/main_code/config_128.yaml', help='Whether to start fresh run or continue previous run')
    add_arg('--seed','-s',  type=str, default='random', help='Seed for random number sequence')
    add_arg('--batchsize','-b',  type=int, default=64, help='batchsize')
    add_arg('--specloss','-spl',action='store_true', help='Whether to use spectral loss')
    return parser.parse_args()


if __name__=='__main__':
    
    ###############
    ### Set up ###
    ###############
   
    ## Read arguments
    args=f_parse_args()
    print('Args',args)
    ngpu,num_epochs=args.ngpu,args.epochs
    mode=args.mode
    batch_size=args.batchsize
    
    # Read config file
    config_file=args.config
    config_dict=f_load_config(config_file)
    print(config_dict)
    
    # Initilize variables
    workers=config_dict['training']['workers']
    nc=config_dict['training']['nc']
    nc,nz,ngf,ndf=config_dict['training']['nc'],config_dict['training']['nz'],config_dict['training']['ngf'],config_dict['training']['ndf']
    lr,beta1=config_dict['training']['lr'],config_dict['training']['beta1']
    kernel_size,stride=config_dict['training']['kernel_size'],config_dict['training']['stride']
    g_padding,d_padding=config_dict['training']['g_padding'],config_dict['training']['d_padding']
    image_size=config_dict['training']['image_size']

    ip_fname=config_dict['data']['ip_fname']
    op_loc=config_dict['data']['op_loc']
    
    ### Initialize 
    if args.seed=='random': manualSeed = np.random.randint(1, 10000)
    else: manualSeed=int(args.seed)
    print("Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device)

    ###############
    ### Load data 
    ###############
 
    img=np.load(ip_fname)[:200000].transpose(0,1,2,3)
    t_img=torch.from_numpy(img)
    print(img.shape,t_img.shape)

    dataset=TensorDataset(torch.Tensor(img))
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=1,drop_last=True)
    print(len(dataset),(dataset[0][0]).shape)



    ### Build Models ###
    # Create generator
    netG = Generator(ngpu,nz,nc,ngf,kernel_size,stride,g_padding).to(device)
    # Apply the weights_init function to randomly initialize all weights
    netG.apply(weights_init)
    print(netG)

    # Create Discriminator
    netD = Discriminator(ngpu, nz,nc,ndf,kernel_size,stride,g_padding).to(device)
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)
    print(netD)
    
    # Handle multi-gpu if desired
    ngpu=torch.cuda.device_count()
    
    print("Number of GPUs used",ngpu)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))
    
    ### Initialize losses

    # Initialize BCELoss function
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(batch_size, 1, 1, nz, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    ### Precompute metrics for input data for computing losses

    ## Stored mean and std of spectrum for full input data once
    mean_spec_data,sdev_spec_data=f_torch_image_spectrum(t_img[:1000],1)
    hist_data=torch.histc(t_img[:1000],bins=50)

    # Create prefix for foldername 
    now=datetime.now()
    fldr_name=now.strftime('%Y%m%d_%H%M%S') ## time format
    # print(fldr_name)
    save_dir=op_loc+fldr_name

    if not os.path.exists(save_dir):
        os.makedirs(save_dir+'/models')
        os.makedirs(save_dir+'/images')

    keys=['Dreal','Dfake','Dfull','G','spec_chi','hist_chi']
    size=len(dataset)/batch_size * num_epochs
    metric_dict=dict(zip(keys,[np.empty((int(np.ceil(size))))*np.nan for i in range(len(keys))]))

    ##############
    ### Train models ###
    # Lists to keep track of progress

    iters = 0
    best_chi1,best_chi2=1e10,1e10

#     ## For checkpointed runs, 
#     if args.mode=='continue':
        
#     elif args.mode=='fresh': 
    
    
    
    t0=time.time()
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        for count, data in enumerate(dataloader, 0):  # For each batch in the dataloader
            tme1=time.time()
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, 1, 1, nz, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Add spectral loss
            mean,sdev=f_torch_image_spectrum(fake,1)  ### compute spectral mean,std for fake images for batch
            
            if args.specloss: # Add spectral loss 
                spec_loss=loss_spectrum(mean,mean_spec_data,sdev,sdev_spec_data,image_size)
                # Combine losses
                errG+=spec_loss
            else: spec_loss=0
            
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Histogram pixel intensity metric
            hist_metric=loss_hist(fake,hist_data.to(device))
            
            tme2=time.time()
            
            # Output training stats
            if count % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, count, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)),
                print("Spec loss: %s,\t hist_chi: %s"%(spec_loss,hist_metric)),
                print("Time taken for step %s : %s"%(iters, tme2-tme1))
            
            # Save metrics
            for key,val in zip(['Dreal','Dfake','Dfull','G','spec_chi','hist_chi'],[errD_real.item(),errD_fake.item(),errD.item(),errG.item(),spec_loss,hist_metric]):
                metric_dict[key][iters]=val

            ### Checkpoint the best model
            checkpoint=True
            if (checkpoint and epoch > 3):
                # Choose best models by metric
                if hist_metric< best_chi1:
                    torch.save({'epoch':epoch,'iters':iters,'G_state':netG.state_dict(),'D_state':netD.state_dict(),
                   'optimizerG_state_dict':optimizerG.state_dict(),'optimizerD_state_dict':optimizerD.state_dict()}
                  , save_dir+'/models/checkpoint_best_hist.tar')
                    best_chi1=hist_metric

                if  (args.specloss and spec_loss< best_chi2):
                    torch.save({'epoch':epoch,'iters':iters,'G_state':netG.state_dict(),'D_state':netD.state_dict(),
                   'optimizerG_state_dict':optimizerG.state_dict(),'optimizerD_state_dict':optimizerD.state_dict()}
                  , save_dir+'/models/checkpoint_best_spec.tar')
                    best_chi2=spec_loss


            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 50 == 0) or ((epoch == num_epochs-1) and (count == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    img_arr=np.array(fake[:,0,:,:])
                    fname='gen_img_epoch-%s_step-%s'%(epoch,iters)
                    np.save(save_dir+'/images/'+fname,img_arr)

            iters += 1
    
    tf=time.time()
    print("Total time",tf-t0)

    ### Save Losses to files
    with open (save_dir+'/metrics.pickle', 'wb') as f:
        pickle.dump(metric_dict,f)

    ### Generate images for best saved models
    ### Load model 
    
    ip_fname=save_dir+'/models/checkpoint_best_spec.tar'
    f_gen_images(netG,optimizerG,nz,device,ip_fname,'spec',save_dir,500)
    
    ip_fname=save_dir+'/models/checkpoint_best_hist.tar'
    f_gen_images(netG,optimizerG,nz,device,ip_fname,'hist',save_dir,500)    