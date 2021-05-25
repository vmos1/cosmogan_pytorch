import torch
import torch.nn as nn
import torch.nn.parallel
import yaml
import numpy as np
import collections

### Transformation functions for image pixel values
def f_transform(x):
    return 2.*x/(x + 4.) - 1.

def f_invtransform(s):
    return 4.*(1. + s)/(1. - s)

# Generator Code
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

def f_gen_images(gdict,netG,optimizerG,sigma,ip_fname,op_loc,op_strg='inf_img_',op_size=500):
    '''Generate images for best saved models
     Arguments: gdict, netG, optimizerG, sigma (parameter value),
                 ip_fname: name of input file
                op_strg: [string name for output file]
                op_size: Number of images to generate
    '''

    nz,device=gdict['nz'],gdict['device']

    try:# handling cpu vs gpu
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
    noise = torch.randn(op_size, 1, 1, 1, nz, device=device) ## Mod for 3D
    tnsr_cosm_params=(torch.ones(op_size,device=device)*sigma).view(op_size,1)
    
    # Generate fake image batch with G
    netG.eval() ## This is required before running inference
    with torch.no_grad(): ## This is important. fails without it for multi-gpu
        gen = netG(noise,tnsr_cosm_params)
        gen_images=gen.detach().cpu().numpy()
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
        print("Error loading saved checkpoint",ip_fname)
        print(e)
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
    
    return iters,epoch,best_chi1,best_chi2,netD,optimizerD,netG,optimizerG


# Mod for 3D
def f_get_model(model_name,gdict):
    ''' Module to define Generator and Discriminator'''
    print("Model name",model_name)
    
    if model_name==2: #### Concatenate sigma input
        class Generator(nn.Module):
            def __init__(self, gdict):
                super(Generator, self).__init__()

                ## Define new variables from dict
                keys=['ngpu','nz','nc','ngf','kernel_size','stride','g_padding']
                ngpu, nz,nc,ngf,kernel_size,stride,g_padding=list(collections.OrderedDict({key:gdict[key] for key in keys}).values())

                self.main = nn.Sequential(
                    # nn.ConvTranspose3d(in_channels, out_channels, kernel_size,stride,padding,output_padding,groups,bias, Dilation,padding_mode)
                    nn.Linear(nz+1,nc*ngf*8**3),# 262144
                    nn.BatchNorm3d(nc,eps=1e-05, momentum=0.9, affine=True),
                    nn.ReLU(inplace=True),
                    View(shape=[-1,ngf*8,4,4,4]),
                    nn.ConvTranspose3d(ngf * 8, ngf * 4, kernel_size, stride, g_padding, output_padding=1, bias=False),
                    nn.BatchNorm3d(ngf*4,eps=1e-05, momentum=0.9, affine=True),
                    nn.ReLU(inplace=True),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose3d( ngf * 4, ngf * 2, kernel_size, stride, g_padding, 1, bias=False),
                    nn.BatchNorm3d(ngf*2,eps=1e-05, momentum=0.9, affine=True),
                    nn.ReLU(inplace=True),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose3d( ngf * 2, ngf, kernel_size, stride, g_padding, 1, bias=False),
                    nn.BatchNorm3d(ngf,eps=1e-05, momentum=0.9, affine=True),
                    nn.ReLU(inplace=True),
                    # state size. (ngf) x 32 x 32
                    nn.ConvTranspose3d( ngf, nc, kernel_size, stride,g_padding, 1, bias=False),
                    nn.Tanh()
                )

            def forward(self, noise,labels):
                x=labels.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
                gen_input=torch.cat((noise,x),-1)
                img=self.main(gen_input)

                return img

        class Discriminator(nn.Module):
            def __init__(self, gdict):
                super(Discriminator, self).__init__()

                ## Define new variables from dict
                keys=['ngpu','nz','nc','ndf','kernel_size','stride','d_padding']
                ngpu, nz,nc,ndf,kernel_size,stride,d_padding=list(collections.OrderedDict({key:gdict[key] for key in keys}).values())

                self.linear_transf=nn.Linear(4,4)
                self.main = nn.Sequential(
                    # input is (nc) x 64 x 64
                    # nn.Conv3d(in_channels, out_channels, kernel_size,stride,padding,output_padding,groups,bias, Dilation,padding_mode)
                    nn.Conv3d(nc+1, ndf,kernel_size, stride, d_padding,  bias=True),
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

            def forward(self, img,labels):
                img_size=gdict['image_size']
                x=labels.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1,4).float() # get to size (batch,1,1,1,4)
                x=self.linear_transf(x)
                x=torch.repeat_interleave(x,int((img_size*img_size*img_size)/4)) # get to size (batch* img^3)
                x=x.view(labels.size(0),1,img_size,img_size,img_size) ## Get to size (batch,1,img,img,img)
                
                ip=torch.cat((img,x),axis=1)
                
                results=[ip]
                lst_idx=[]
                for i,submodel in enumerate(self.main.children()):
                    mid_output=submodel(results[-1])
                    results.append(mid_output)
                    ## Select indices in list corresponding to output of Conv layers
                    if submodel.__class__.__name__.startswith('Conv'):
        #                 print(submodel.__class__.__name__)
        #                 print(mid_output.shape)
                        lst_idx.append(i)

                FMloss=True
                if FMloss:
                    ans=[results[1:][i] for i in lst_idx + [-1]]
                else :
                    ans=results[-1]
                
                return ans                
                

    elif model_name==3:#### Model 3: with ConditionalInstanceNorm2d,
        class ConditionalInstanceNorm2d(nn.Module):
            def __init__(self, num_features, num_params):
                super().__init__()
                self.num_features = num_features
                self.InstNorm = nn.InstanceNorm2d(num_features, affine=False)
                self.affine = nn.Linear(num_params, num_features * 2)
                self.affine.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
                self.affine.weight.data[:, num_features:].zero_()  # Initialise bias at 0

            def forward(self, x, y):
                out = self.InstNorm(x)
                gamma, beta = self.affine(y).chunk(2, 1)
                out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
                return out

        class ConditionalSequential(nn.Sequential):
            def __init__(self,*args):
                super(ConditionalSequential, self).__init__(*args)

            def forward(self, inputs, labels):
                for module in self:
                    if module.__class__ is ConditionalInstanceNorm2d:
                        inputs = module(inputs, labels.float())
                    else:
                        inputs = module(inputs)

                return inputs

        class Generator(nn.Module):
            def __init__(self, gdict):
                super(Generator, self).__init__()

                ## Define new variables from dict
                keys=['ngpu','nz','nc','ngf','kernel_size','stride','g_padding']
                ngpu, nz,nc,ngf,kernel_size,stride,g_padding=list(collections.OrderedDict({key:gdict[key] for key in keys}).values())

                self.main = ConditionalSequential(
                    # nn.ConvTranspose3d(in_channels, out_channels, kernel_size,stride,padding,output_padding,groups,bias, Dilation,padding_mode)
                    nn.Linear(nz,nc*ngf*8**3),# 262144
                    nn.BatchNorm3d(nc,eps=1e-05, momentum=0.9, affine=True),
                    nn.ReLU(inplace=True),
                    View(shape=[-1,ngf*8,4,4,4]),
                    nn.ConvTranspose3d(ngf * 8, ngf * 4, kernel_size, stride, g_padding, output_padding=1, bias=False),
                    ConditionalInstanceNorm2d(ngf*4,1),
                    nn.ReLU(inplace=True),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose3d( ngf * 4, ngf * 2, kernel_size, stride, g_padding, 1, bias=False),
                    ConditionalInstanceNorm2d(ngf*2,1),
                    nn.ReLU(inplace=True),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose3d( ngf * 2, ngf, kernel_size, stride, g_padding, 1, bias=False),
                    ConditionalInstanceNorm2d(ngf,1),
                    nn.ReLU(inplace=True),
                    # state size. (ngf) x 32 x 32
                    nn.ConvTranspose3d( ngf, nc, kernel_size, stride,g_padding, 1, bias=False),
                    nn.Tanh()
                )
                
                
            def forward(self, noise,labels):
                img=self.main(noise,labels)

                return img

        class Discriminator(nn.Module):
            def __init__(self, gdict):
                super(Discriminator, self).__init__()

                ## Define new variables from dict
                keys=['ngpu','nz','nc','ndf','kernel_size','stride','d_padding']
                ngpu, nz,nc,ndf,kernel_size,stride,d_padding=list(collections.OrderedDict({key:gdict[key] for key in keys}).values())
                
                self.main = nn.Sequential(
                    # input is (nc) x 64 x 64
                    # nn.Conv3d(in_channels, out_channels, kernel_size,stride,padding,output_padding,groups,bias, Dilation,padding_mode)
                    nn.Conv3d(nc, ndf,kernel_size, stride, d_padding,  bias=True),
                    ConditionalInstanceNorm2d(ndf,1),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf) x 32 x 32
                    nn.Conv3d(ndf, ndf * 2, kernel_size, stride, d_padding, bias=True),
                    ConditionalInstanceNorm2d(ndf*2,1),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*2) x 16 x 16
                    nn.Conv3d(ndf * 2, ndf * 4, kernel_size, stride, d_padding, bias=True),
                    ConditionalInstanceNorm2d(ndf*4,1),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*4) x 8 x 8
                    nn.Conv3d(ndf * 4, ndf * 8, kernel_size, stride, d_padding, bias=True),
                    ConditionalInstanceNorm2d(ndf*8,1),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*8) x 4 x 4
                    nn.Flatten(),
                    nn.Linear(nc*ndf*8*8*8, 1)
        #             nn.Sigmoid()
                )
        
            def forward(self, ip,labels):   
                results=[ip]
                lst_idx=[]
                for i,submodel in enumerate(self.main.children()):
                    mid_output=submodel(results[-1])
                    results.append(mid_output)
                    ## Select indices in list corresponding to output of Conv layers
                    if submodel.__class__.__name__.startswith('Conv'):
        #                 print(submodel.__class__.__name__)
        #                 print(mid_output.shape)
                        lst_idx.append(i)

                FMloss=True
                if FMloss:
                    ans=[results[1:][i] for i in lst_idx + [-1]]
                else :
                    ans=results[-1]
                return ans

    return Generator, Discriminator