# Introduction
This repository contains code to implement Generative Adversarial Neural networks to produce images of matter distribution in the universe for different sets of cosmological parameters. Dataset consisits of N-body cosmology simulation maps obtained from PyCOLA. The pixel values represent matter density.

The aim is to build conditional GANs to produce images for different classes corresponding to different sets of cosmological parameters.
# Plots

## 3D GAN results
We develop a simple GAN trained on 3D images of size 64^3.
### Metric comparison
3D GAN: Pixel intensity | 3D GAN: Power spectrum |
:-------------:|:---------------:
![Pixel intensity](https://github.com/vmos1/cosmogan_pytorch/blob/master/images/3d_hist_best.png)| ![Power spectrum](https://github.com/vmos1/cosmogan_pytorch/blob/master/images/3d_spec_best.png)

### Image comparison
Below are 2D snapshots of a set of 3D images. The input images are to the left and the GAN generated images are to the right.
3D GAN: Input images | 3D GAN: Generated images |
:-------------:|:---------------:
![2D slices of input images](https://github.com/vmos1/cosmogan_pytorch/blob/master/images/3d_reference.png)| ![2D slices of generated images](https://github.com/vmos1/cosmogan_pytorch/blob/master/images/3d_generated.png)


## 2D conditional GAN results
We develop a conditional GAN trained on 2D images of size $128^2$ for 3 different values of the cosmological paramter sigma.
### Metric comparison
2D cGAN: Pixel intensity | 2D cGAN: Power spectrum |
:-------------:|:---------------:
![Pixel intensity](https://github.com/vmos1/cosmogan_pytorch/blob/master/images/2d_cgan_hist_best.png) |![Power spectrum](https://github.com/vmos1/cosmogan_pytorch/blob/master/images/2d_cgan_spec_best.png)

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
| [1_basic_GAN/1_main_code](https://github.com/vmos1/cosmogan_pytorch/tree/master/code/1_basic_GAN/1_main_code) | Folder containing main training code |
| [1_basic_GAN/launch_train_pytorch.ipynb](https://github.com/vmos1/cosmogan_pytorch/blob/master/code/1_basic_GAN/launch_train_pytorch.ipynb) |  Notebook that launches script to run training |
| [1_basic_GAN/launch_compute_chisqr.ipynb](https://github.com/vmos1/cosmogan_pytorch/blob/master/code/1_basic_GAN/launch_compute_chisqr.ipynb)| Notebook that launches script to run post-run metric computation |
|[1_basic_GAN/2_analysis/1_pytorch_analyze-results.ipynb](https://github.com/vmos1/cosmogan_pytorch/blob/master/code/1_basic_GAN/2_analysis/1_pytorch_analyze-results.ipynb) | Notebook to analyze GAN results and view best epoch-steps |
