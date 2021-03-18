# Introduction
This repository contains code to implement Generative Adversarial Neural networks to produce images of matter distribution in the universe for different sets of cosmological parameters. Dataset consisits of N-body cosmology simulation maps obtained from PyCOLA.

The aim is to build conditional GANs to produce images for different classes corresponding to different sets of cosmological parameters.

# Repository information
There separate directories to run GAN and cGAN in for both 2D and 3D datasets.
The Table below describes the content of various folders in this repository.

| Description | Location |
| --- | ---|
| 2d GAN - Main code | code/1_basic_GAN/1_main_code/ |
| 2d GAN - Analysis code | code/1_basic_GAN/2_analysis |
| 2d CGAN - code | code/3_cond_GAN/1_main_code |
| 2d CGAN - analysis | code/3_cond_GAN/2_cgan_analysis/ |
| 3d GAN | code/4_basic_3d_GAN/1_main_code/ |
| 3d CGAN - code | code/5_3d_cgan/1_main_code |


Each folder contains a jupyter notebook to quickly test the code, a folder with the full code, a launch script to run the code on cori GPUs at NERSC, a script to perform post-run computation of metrics for different stored images and a folder with analysis codes to inspect the performance of the code. Below is an example for 2D GAN
| Name | Description |
| --- | ---|
| [1_basic_GAN/cosmogan_train.ipynb](https://github.com/vmos1/cosmogan_pytorch/blob/master/code/1_basic_GAN/cosmogan_train.ipynb) | Jupyter notebook to test GAN |
| [1_main_code](https://github.com/vmos1/cosmogan_pytorch/tree/master/code/1_basic_GAN/1_main_code) | Folder containing main training code |
| [launch_train_pytorch.ipynb](https://github.com/vmos1/cosmogan_pytorch/blob/master/code/1_basic_GAN/launch_train_pytorch.ipynb) |  Notebook that launches script to run training |
| [launch_compute_chisqr.ipynb](https://github.com/vmos1/cosmogan_pytorch/blob/master/code/1_basic_GAN/launch_compute_chisqr.ipynb)| Notebook that launches script to run post-run metric computation |
|[1_pytorch_analyze-results.ipynb](https://github.com/vmos1/cosmogan_pytorch/blob/master/code/1_basic_GAN/2_analysis/1_pytorch_analyze-results.ipynb) | Notebook to analyze GAN results and view best epoch-steps |
| --- | ---|
