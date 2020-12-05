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
# import random
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

import logging
import collections

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
    add_arg('--batch_size','-b',  type=int, default=64, help='batchsize')
    add_arg('--learn_rate','-lr',  type=float, default=0.0002, help='Learn rate')
    add_arg('--lambda1','-ld1',  type=float, default=1.0, help='Coupling for spectral loss values')
    add_arg('--specloss','-spl',dest='spec_loss_flag',action='store_true', help='Whether to use spectral loss')
    add_arg('--deterministic','-dt',action='store_true', help='Run with deterministic sequence')
    add_arg('--model','-ml',type=int, choices=[1,2,3], help='Model number')
    return parser.parse_args()


def f_train_loop(dataloader,metrics_df,gdict):
    ''' Train single epoch '''
    
    ## Define new variables from dict
    keys=['start_epoch','epochs','iters','best_chi1','best_chi2','save_dir','device','flip_prob','nz','num_classes','batch_size','bns']
    start_epoch,epochs,iters,best_chi1,best_chi2,save_dir,device,flip_prob,nz,num_classes,batch_size,bns=list(collections.OrderedDict({key:gdict[key] for key in keys}).values())
    
    for epoch in range(start_epoch,epochs):
        t_epoch_start=time.time()
        for count, data in enumerate(dataloader, 0):
            
            ####### Train GAN ########
            netG.train(); netD.train();  ### Need to add these after inference and before training

            tme1=time.time()
            ### Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()

            real_cpu = data[0].to(device)
            real_categories=data[1].to(device)
            real_categories=real_categories.squeeze(-1)
            
            b_size = real_cpu.size(0)
            real_label = torch.full((b_size,), 1, device=device)
            fake_label = torch.full((b_size,), 0, device=device)
            g_label = torch.full((b_size,), 1, device=device) # No flipping for Generator labels
            # Flip labels with probability flip_prob for Discriminator
            for idx in np.random.choice(np.arange(b_size),size=int(np.ceil(b_size*flip_prob))):
                real_label[idx]=0; fake_label[idx]=1
            
            # Generate fake image batch with G
            noise = torch.randn(b_size, 1, 1, nz, device=device)
#             fake_categories=torch.randint(num_classes,(1,batch_size),device=device).long().view(batch_size,1,1)
            fake_categories=torch.randint(gdict['num_classes'],(gdict['batch_size'],1),device=gdict['device'])
            if gdict['model']!=3: fake_categories=fake_categories.long().unsqueeze(1)
            fake = netG(noise,fake_categories)  
            
            # Forward pass real batch through D
            output = netD(real_cpu,real_categories).view(-1)
            errD_real = criterion(output, real_label)
            errD_real.backward()
            D_x = output.mean().item()
            
            # Forward pass fake batch through D
            output = netD(fake.detach(),fake_categories).view(-1)
            errD_fake = criterion(output, fake_label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            
            ###Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            output = netD(fake,fake_categories).view(-1)
            errG_adv = criterion(output, g_label)
            # Histogram pixel intensity loss
            
#             hist_gen=f_compute_hist(fake,bins=bns)
#             hist_loss=loss_hist(hist_gen,hist_val.to(device))
            hist_loss=f_get_hist_cond(fake,fake_categories,bns,gdict,hist_val_tnsr)
            
            # Add spectral loss
#             mean,sdev=f_torch_image_spectrum(f_invtransform(fake),1,r.to(device),ind.to(device))
#             spec_loss=loss_spectrum(mean,mean_spec_val.to(device),sdev,sdev_spec_val.to(device),image_size)
#             logging.info(fake.device,fake_categories.device)
            spec_loss=f_get_spec_cond(fake,fake_categories,gdict,spec_mean_tnsr,spec_sdev_tnsr,r,ind)

            if gdict['spec_loss_flag']: errG=errG_adv+spec_loss
            else: errG=errG_adv
#             errG=errG_adv
            if torch.isnan(errG).any():
                logging.info(errG)
                raise SystemError
            
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            tme2=time.time()
            
            ####### Store metrics ########
            # Output training stats
            if count % gdict['checkpoint_size'] == 0:
                logging.info('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_adv: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, epochs, count, len(dataloader), errD.item(), errG_adv.item(),errG.item(), D_x, D_G_z1, D_G_z2)),
                logging.info("Spec loss: %s,\t hist loss: %s"%(spec_loss.item(),hist_loss.item())),
                logging.info("Training time for step %s : %s"%(iters, tme2-tme1))

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
                fake = netG(fixed_noise,fixed_categories)
#                 hist_gen=f_compute_hist(fake,bins=bns)
#                 hist_chi=loss_hist(hist_gen,hist_val.to(device))
#                 mean,sdev=f_torch_image_spectrum(f_invtransform(fake),1,r.to(device),ind.to(device))
#                 spec_chi=loss_spectrum(mean,mean_spec_val.to(device),sdev,sdev_spec_val.to(device),image_size)      
                hist_chi=f_get_hist_cond(fake,fixed_categories,bns,gdict,hist_val_tnsr)
                spec_chi=f_get_spec_cond(fake,fixed_categories,gdict,spec_mean_tnsr,spec_sdev_tnsr,r,ind)

            # Storing chi for next step
            for col,val in zip(['spec_chi','hist_chi'],[spec_chi.item(),hist_chi.item()]):  metrics_df.loc[iters,col]=val            

            # Checkpoint model for continuing run
            if count == len(dataloader)-1: ## Check point at last step of epoch
                f_save_checkpoint(epoch,iters,best_chi1,best_chi2,netG,netD,optimizerG,optimizerD,save_loc=save_dir+'/models/checkpoint_last.tar')  

            if (checkpoint and (epoch > 1)): # Choose best models by metric
                if hist_chi< best_chi1:
                    f_save_checkpoint(epoch,iters,best_chi1,best_chi2,netG,netD,optimizerG,optimizerD,save_loc=save_dir+'/models/checkpoint_best_hist.tar')
                    best_chi1=hist_chi.item()
                    logging.info("Saving best hist model at epoch %s, step %s."%(epoch,iters))

                if  spec_chi< best_chi2:
                    f_save_checkpoint(epoch,iters,best_chi1,best_chi2,netG,netD,optimizerG,optimizerD,save_loc=save_dir+'/models/checkpoint_best_spec.tar')
                    best_chi2=spec_chi.item()
                    logging.info("Saving best spec model at epoch %s, step %s"%(epoch,iters))

            # Save G's output on fixed_noise
            if ((iters % gdict['checkpoint_size'] == 0) or ((epoch == epochs-1) and (count == len(dataloader)-1))):
                netG.eval()
                with torch.no_grad():
                    for category in range(num_classes):
#                         cat_tensor=torch.LongTensor(np.ones(batch_size)*category).view(batch_size,1,1)
                        tnsr_categories=(torch.ones(batch_size,device=device)*category).view(batch_size,1)
                        if gdict['model']!=3: tnsr_categories=tnsr_categories.long().unsqueeze(1)
                        fake = netG(fixed_noise,tnsr_categories).detach().cpu()
                        img_arr=np.array(fake[:,0,:,:])
                        fname='gen_img_label-%s_epoch-%s_step-%s'%(category,epoch,iters)
                        np.save(save_dir+'/images/'+fname,img_arr)
        
        t_epoch_end=time.time()
        logging.info("Time taken for epoch %s: %s"%(epoch,t_epoch_end-t_epoch_start))
        # Save Metrics to file after each epoch
        metrics_df.to_pickle(save_dir+'/df_metrics.pkle')
        
    logging.info("best chis: {0}, {1}".format(best_chi1,best_chi2))


def f_init_gdict(gdict,config_dict):
    ''' Initialize the global dictionary gdict with values in config file'''
    keys1=['workers','nc','nz','ngf','ndf','beta1','kernel_size','stride','g_padding','d_padding','flip_prob']
    keys2=['image_size','checkpoint_size','num_imgs','ip_fname','op_loc']
    for key in keys1: gdict[key]=config_dict['training'][key]
    for key in keys2: gdict[key]=config_dict['data'][key]
    
#########################
### Main code #######
#########################

if __name__=="__main__":
    torch.backends.cudnn.benchmark=True
#     torch.backends.cudnn.deterministic=True
    
    t0=time.time()
    #################################
    args=f_parse_args()
    
    ### Set up ###
    config_file=args.config
    config_dict=f_load_config(config_file)

    # Initilize variables    
    gdict={}
    f_init_gdict(gdict,config_dict)
    
    ## Add args variables to gdict
    for key in ['ngpu','batch_size','mode','spec_loss_flag','epochs','learn_rate','lambda1','model']:
        gdict[key]=vars(args)[key]
       
    ###### Set up directories #######
    if gdict['mode']=='fresh':
        # Create prefix for foldername        
        fldr_name=datetime.now().strftime('%Y%m%d_%H%M%S') ## time format
        gdict['save_dir']=gdict['op_loc']+fldr_name+'_'+args.run_suffix
        
        if not os.path.exists(gdict['save_dir']):
            os.makedirs(gdict['save_dir']+'/models')
            os.makedirs(gdict['save_dir']+'/images')
        ### Initialize variables      
        best_chi1,best_chi2,iters,start_epoch=1e10,1e10,0,0
        
    elif args.mode=='continue': ## For checkpointed runs
        gdict['save_dir']=args.ip_fldr
        
        iters,start_epoch,best_chi1,best_chi2=f_load_checkpoint(save_dir+'/models/checkpoint_last.tar',netG,netD,optimizerG,optimizerD) 

        logging.info("Continuing existing run. Loading checkpoint with epoch {0} and step {1}".format(start_epoch,iters))
        start_epoch+=1  ## Start with the next epoch   
        ### Read loss data
        with open (save_dir+'/metrics.pickle','rb') as f:
            metrics_dict=pickle.load(f) 
    
    ## Add to gdict
    for key,val in zip(['best_chi1','best_chi2','iters','start_epoch'],[best_chi1,best_chi2,iters,start_epoch]): gdict[key]=val

    ### Write all logging.info statements to stdout and log file (different for jpt notebooks)
    logfile=gdict['save_dir']+'/log.log'
    logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    
    Lg = logging.getLogger()
    Lg.setLevel(logging.DEBUG)
    lg_handler_file = logging.FileHandler(logfile)
    lg_handler_stdout = logging.StreamHandler(sys.stdout)
    Lg.addHandler(lg_handler_file)
    Lg.addHandler(lg_handler_stdout)
    
    logging.info('Args: {0}'.format(args))
    logging.info(config_dict)
    logging.info('Start: %s'%(datetime.now().strftime('%Y-%m-%d  %H:%M:%S')))
    if gdict['spec_loss_flag']: logging.info("Using Spectral loss")

    ### Override (different for jpt notebooks)
#     gdict['num_imgs']=200
    
    ## Special declarations
    gdict['bns']=50
    gdict['num_classes']=4
    gdict['device']=torch.device("cuda" if (torch.cuda.is_available() and gdict['ngpu'] > 0) else "cpu")
    gdict['ngpu']=torch.cuda.device_count()
    logging.info(gdict)
    
    ### Initialize random seed
    if args.seed=='random': manualSeed = np.random.randint(1, 10000)
    else: manualSeed=int(args.seed)
    logging.info("Seed:{0}".format(manualSeed))
#     random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    logging.info('Device:{0}'.format(gdict['device']))

    if args.deterministic: 
        logging.info("Running with deterministic sequence. Performancd will be slower")
        torch.backends.cudnn.deterministic=True
    
    #################################   
    ### Read data from different files
    sigma_list=[0.5,0.65,0.8,1.1]
    for count,sigma in enumerate(sigma_list):
        fname=gdict['ip_fname']+'/norm_1_sig_%s_train_val.npy'%(sigma)
        x=np.load(fname,mmap_mode='r')[:gdict['num_imgs']].transpose(0,1,2,3)
        size=x.shape[0]
        y=count*np.ones(size)

        if count ==0:
            img=x[:]
            lab=y[:]
        else: 
            img=np.vstack([img,x]) # Store images
            lab=np.hstack([lab,y]) # Store class labels

    t_img=torch.from_numpy(img)
    labels=torch.LongTensor(lab).view(size*4,1,1)
    logging.info("%s, %s"%(labels.shape,t_img.shape))
    
    dataset=TensorDataset(t_img,labels)
    dataloader=DataLoader(dataset,batch_size=gdict['batch_size'],shuffle=True,num_workers=1,drop_last=True)

    # Precompute metrics with validation data for computing losses
    with torch.no_grad():
        spec_mean_list=[];spec_sdev_list=[];hist_val_list=[]
        
        for count,sigma in enumerate(sigma_list):
            ip_fname=gdict['ip_fname']+'/norm_1_sig_%s_train_val.npy'%(sigma)
            val_img=np.load(ip_fname,mmap_mode='r')[-3000:].transpose(0,1,2,3)
            t_val_img=torch.from_numpy(val_img).to(gdict['device'])

            # Precompute radial coordinates
            if count==0: 
                r,ind=f_get_rad(img)
                r=r.to(gdict['device']); ind=ind.to(gdict['device'])
            # Stored mean and std of spectrum for full input data once
            mean_spec_val,sdev_spec_val=f_torch_image_spectrum(f_invtransform(t_val_img),1,r,ind)
            hist_val=f_compute_hist(t_val_img,bins=gdict['bns'])
            
            spec_mean_list.append(mean_spec_val)
            spec_sdev_list.append(sdev_spec_val)
            hist_val_list.append(hist_val)
        spec_mean_tnsr=torch.stack(spec_mean_list)
        spec_sdev_tnsr=torch.stack(spec_sdev_list)
        hist_val_tnsr=torch.stack(hist_val_list)
        
        del val_img; del t_val_img; del img; del t_img; del spec_mean_list; del spec_sdev_list; del hist_val_list
    
    #################################
    ###### Build Networks ###
    # Define Models
    Generator, Discriminator=f_get_model(gdict['model'],gdict)
    logging.info("Building GAN networks")
    # Create Generator
    netG = Generator(gdict).to(gdict['device'])
    netG.apply(weights_init)
#     logging.info(netG)
#     summary(netG,(1,1,64))
    # Create Discriminator
    netD = Discriminator(gdict).to(gdict['device'])
    netD.apply(weights_init)
#     logging.info(netD)
#     summary(netD,(1,128,128))
    
    logging.info("Number of GPUs used %s"%(gdict['ngpu']))
    if (gdict['device'].type == 'cuda') and (gdict['ngpu'] > 1):
        netG = nn.DataParallel(netG, list(range(gdict['ngpu'])))
        netD = nn.DataParallel(netD, list(range(gdict['ngpu'])))

    # Initialize BCELoss function
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=gdict['learn_rate'], betas=(gdict['beta1'], 0.999),eps=1e-7)
    optimizerG = optim.Adam(netG.parameters(), lr=gdict['learn_rate'], betas=(gdict['beta1'], 0.999),eps=1e-7)
    
    fixed_noise = torch.randn(gdict['batch_size'], 1, 1, gdict['nz'], device=gdict['device']) #Latent vectors to view G progress
#     fixed_categories=torch.randint(gdict['num_classes'],(1,gdict['batch_size']),device=gdict['device']).long().view(gdict['batch_size'],1,1)
    fixed_categories=torch.randint(gdict['num_classes'],(gdict['batch_size'],1),device=gdict['device'])
    if gdict['model']!=3: fixed_categories=fixed_categories.long().unsqueeze(1)

    #################################       
    ### Set up metrics dataframe
    cols=['step','epoch','Dreal','Dfake','Dfull','G_adv','G_full','spec_loss','hist_loss','spec_chi','hist_chi','D(x)','D_G_z1','D_G_z2','time']
    # size=int(len(dataloader) * epochs)+1
    metrics_df=pd.DataFrame(columns=cols)

    #################################
    ########## Train loop and save metrics and images ######
    logging.info("Starting Training Loop...")
    f_train_loop(dataloader,metrics_df,gdict)

    ## Generate images for best saved models ######
    for cl in np.arange(gdict['num_classes']):
        model_fname=gdict['save_dir']+'/models/checkpoint_best_spec.tar'
        f_gen_images(gdict,netG,optimizerG,cl,model_fname,'spec',2000)

        model_fname=gdict['save_dir']+'/models/checkpoint_best_hist.tar'
        f_gen_images(gdict,netG,optimizerG,cl,model_fname,'hist',2000)

    tf=time.time()
    logging.info("Total time %s"%(tf-t0))
    logging.info('End: %s'%(datetime.now().strftime('%Y-%m-%d  %H:%M:%S')))
