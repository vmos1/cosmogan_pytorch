import torch
import torch.nn as nn
import torch.nn.parallel
import yaml
import numpy as np

def f_load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Generator(nn.Module):
    def __init__(self, ngpu,nz,nc,ngf,kernel_size,stride,g_padding):
        super(Generator, self).__init__()
        self.ngpu = ngpu
#         self.nz,self.nc,self.ngf=nz,nc,ngf
#         self.kernel_size,self.g_padding=kernel_size,g_padding

        self.main = nn.Sequential(
            # nn.ConvTranspose2d(in_channels, out_channels, kernel_size,stride,padding,output_padding,groups,bias, Dilation,padding_mode)
            nn.Linear(nz,nc*ngf*8*8*8),# 32768
            nn.BatchNorm2d(nc),
            nn.ReLU(True),
            View(shape=[-1,ngf*8,8,8]),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size, stride, g_padding, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, kernel_size, stride, g_padding, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, kernel_size, stride, g_padding, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, kernel_size, stride,g_padding, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    
    def forward(self, input):
        return self.main(input)


# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, ngpu, nz,nc,ndf,kernel_size,stride,d_padding):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            # nn.Conv2d(in_channels, out_channels, kernel_size,stride,padding,output_padding,groups,bias, Dilation,padding_mode)
            nn.Conv2d(nc, ndf,kernel_size, stride, d_padding,  bias=False),
            nn.BatchNorm2d(ndf,eps=1e-05, momentum=0.9, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, kernel_size, stride, d_padding, bias=False),
            nn.BatchNorm2d(ndf * 2,eps=1e-05, momentum=0.9, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size, stride, d_padding, bias=False),
            nn.BatchNorm2d(ndf * 4,eps=1e-05, momentum=0.9, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size, stride, d_padding, bias=False),
            nn.BatchNorm2d(ndf * 8,eps=1e-05, momentum=0.9, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Flatten(),
            nn.Linear(nc*ndf*8*8*8, 1)
        )

    def forward(self, input):
        return self.main(input)

def f_gen_images(netG,optimizerG,nz,device,ip_fname,strg,save_dir,op_size=500):
    '''Generate images for best saved models
     Arguments: ip_fname: name of input file
                strg: ['hist' or 'spec']
                op_size: Number of images to generate
    '''

    try:
        checkpoint=torch.load(ip_fname)
    except Exception as e:
        print(e)
        print("skipping generation of images for ",ip_fname)
        return
    
    netG.load_state_dict(checkpoint['G_state'])
#    netD.load_state_dict(checkpoint['D_state'])

    ## Load other stuff
    iters=checkpoint['iters']
    epoch=checkpoint['epoch']
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    
    # Generate batch of latent vectors
    noise = torch.randn(op_size, 1, 1, nz, device=device)
    # Generate fake image batch with G
    netG.eval() ## This is required before running inference
    gen = netG(noise)
    gen_images=gen.detach().cpu().numpy()[:,0,:,:]
    print(gen_images.shape)

    op_fname='best-%s_gen_img_epoch-%s_step-%s.npy'%(strg,epoch,iters)

    np.save(save_dir+'/images/'+op_fname,gen_images)

    print("Image saved in ",op_fname)
    
    
def f_save_checkpoint(epoch,iters,best_chi1,best_chi2,netG,netD,optimizerG,optimizerD,save_loc):
    ''' Checkpoint model '''
    torch.save({'epoch':epoch,'iters':iters,'best_chi1':best_chi1,'best_chi2':best_chi2,
                'G_state':netG.state_dict(),'D_state':netD.state_dict(),'optimizerG_state_dict':optimizerG.state_dict(),
                'optimizerD_state_dict':optimizerD.state_dict()}, save_loc) 
    
    
def f_load_checkpoint(ip_fname,netG,netD,optimizerG,optimizerD):
    ''' Load saved checkpoint
    Also loads step, epoch, best_chi1, best_chi2'''
    
    try:
        checkpoint=torch.load(ip_fname)
    except Exception as e:
        print(e)
        print("skipping generation of images for ",ip_fname)
        raise SystemError
    
    ## Load checkpoint
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