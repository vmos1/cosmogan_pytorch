import torch
import torch.nn as nn
import torch.nn.parallel
import yaml
import numpy as np
import collections

def f_load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

### Transformation functions for image pixel values
def f_transform(x):
    return 2.*x/(x + 4.) - 1.

def f_invtransform(s):
    return 4.*(1. + s)/(1. - s)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Generator(nn.Module):
    def __init__(self, gdict):
        super(Generator, self).__init__()

        ## Define new variables from dict
        keys=['ngpu','nz','nc','ngf','kernel_size','stride','g_padding']
        ngpu, nz,nc,ngf,kernel_size,stride,g_padding=list(collections.OrderedDict({key:gdict[key] for key in keys}).values())
        kernel_size=2;g_padding=0;stride=2;
        
        self.main = nn.Sequential(
            # nn.ConvTranspose3d(in_channels, out_channels, kernel_size,stride,padding,output_padding,groups,bias, Dilation,padding_mode)
            nn.Linear(nz,nc*ngf*8**3),# 262144
            nn.BatchNorm3d(nc,eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(inplace=True),
            View(shape=[-1,ngf*8,4,4,4]),
            nn.ConvTranspose3d(ngf * 8, ngf * 4, kernel_size, stride, g_padding, output_padding=0, bias=False),
            nn.BatchNorm3d(ngf*4,eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose3d( ngf * 4, ngf * 2, kernel_size, stride, g_padding, 0, bias=False),
            nn.BatchNorm3d(ngf*2,eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose3d( ngf * 2, ngf, kernel_size, stride, g_padding, 0, bias=False),
            nn.BatchNorm3d(ngf,eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose3d( ngf, nc, kernel_size, stride,g_padding, 0, bias=False),
            nn.Tanh()
        )
    
    def forward(self, ip):
        return self.main(ip)

class Discriminator(nn.Module):
    def __init__(self, gdict):
        super(Discriminator, self).__init__()
        
        ## Define new variables from dict
        keys=['ngpu','nz','nc','ndf','kernel_size','stride','d_padding']
        ngpu, nz,nc,ndf,kernel_size,stride,d_padding=list(collections.OrderedDict({key:gdict[key] for key in keys}).values())        
#         kernel_size=2;d_padding=0;stride=2;

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            # nn.Conv3d(in_channels, out_channels, kernel_size,stride,padding,output_padding,groups,bias, Dilation,padding_mode)
            nn.Conv3d(nc, ndf,kernel_size, stride, d_padding,  bias=True),
            nn.BatchNorm3d(ndf,eps=1e-05, momentum=0.9, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv3d(ndf, ndf * 2, kernel_size, stride, d_padding, bias=True),
            nn.BatchNorm3d(ndf * 2,eps=1e-05, momentum=0.9, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv3d(ndf * 2, ndf * 4, kernel_size, stride, d_padding, bias=True),
            nn.BatchNorm3d(ndf * 4,eps=1e-05, momentum=0.9, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv3d(ndf * 4, ndf * 8, kernel_size, stride, d_padding, bias=True),
            nn.BatchNorm3d(ndf * 8,eps=1e-05, momentum=0.9, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Flatten(),
            nn.Linear(nc*ndf*8*8*8, 1)
#             nn.Sigmoid()
        )

    def forward(self, ip):
        return self.main(ip)


def f_gen_images(gdict,netG,optimizerG,ip_fname,op_loc,op_strg='inf_img_',op_size=500):
    '''Generate images for best saved models
     Arguments: gdict, netG, optimizerG, 
                 ip_fname: name of input file
                op_strg: [string name for output file]
                op_size: Number of images to generate
    '''

    nz,device=gdict['nz'],gdict['device']

    try:
        if torch.cuda.is_available(): checkpoint=torch.load(ip_fname)
        else: checkpoint=torch.load(ip_fname,map_location=torch.device('cpu'))
    except Exception as e:
        print(e)
        print("skipping generation of images for ",ip_fname)
        return
    
    ## Load checkpoint
    if gdict['multi-gpu']:
        netG.module.load_state_dict(checkpoint['G_state'])
    else:
        netG.load_state_dict(checkpoint['G_state'])
    
    ## Load other stuff
    iters=checkpoint['iters']
    epoch=checkpoint['epoch']
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    
    # Generate batch of latent vectors
    noise = torch.randn(op_size, 1, 1, 1, nz, device=device)
    # Generate fake image batch with G
    netG.eval() ## This is required before running inference
    gen = netG(noise)
    gen_images=gen.detach().cpu().numpy()[:,:,:,:]
    print(gen_images.shape)
    
    op_fname='%s_epoch-%s_step-%s.npy'%(op_strg,epoch,iters)

    np.save(op_loc+op_fname,gen_images)

    print("Image saved in ",op_fname)
    
def f_save_checkpoint(gdict,epoch,iters,best_chi1,best_chi2,netG,netD,optimizerG,optimizerD,save_loc):
    ''' Checkpoint model '''
    
    if gdict['multi-gpu']: ## Dataparallel
        torch.save({'epoch':epoch,'iters':iters,'best_chi1':best_chi1,'best_chi2':best_chi2,
                'G_state':netG.module.state_dict(),'D_state':netD.module.state_dict(),'optimizerG_state_dict':optimizerG.state_dict(),
                'optimizerD_state_dict':optimizerD.state_dict()}, save_loc) 
    else :
        torch.save({'epoch':epoch,'iters':iters,'best_chi1':best_chi1,'best_chi2':best_chi2,
                'G_state':netG.state_dict(),'D_state':netD.state_dict(),'optimizerG_state_dict':optimizerG.state_dict(),
                'optimizerD_state_dict':optimizerD.state_dict()}, save_loc)
    
def f_load_checkpoint(ip_fname,netG,netD,optimizerG,optimizerD,gdict):
    ''' Load saved checkpoint
    Also loads step, epoch, best_chi1, best_chi2'''
    
    try:
        checkpoint=torch.load(ip_fname)
    except Exception as e:
        print(e)
        print("skipping generation of images for ",ip_fname)
        raise SystemError
    
    ## Load checkpoint
    if gdict['multi-gpu']:
        netG.module.load_state_dict(checkpoint['G_state'])
        netD.module.load_state_dict(checkpoint['D_state'])
    else:
        netG.load_state_dict(checkpoint['G_state'])
        netD.load_state_dict(checkpoint['D_state'])
    
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    
    iters=checkpoint['iters']
    epoch=checkpoint['epoch']
    best_chi1=checkpoint['best_chi1']
    best_chi2=checkpoint['best_chi2']

    netG.train()
    netD.train()
    
    return iters,epoch,best_chi1,best_chi2
