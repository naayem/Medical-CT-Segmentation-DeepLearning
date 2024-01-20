# Master's Thesis - Deep Learning for Computed Tomography Scans

![image](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![image](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)


## Introduction

This repository is associated with my Master's thesis titled "Deep Learning for Computed Tomography Scans," completed under the supervision of Prof. Alexandre Alahi at the Visual Intelligence for Transportation Laboratory, EPFL. The focus of this thesis was to enhance medical image segmentation using deep learning techniques, with a particular emphasis on the challenging task of segmenting anatomical structures in CT scans.

Medical image segmentation is a critical aspect of medical imaging, with applications in diagnosis, treatment planning, and image-guided surgery. This thesis aimed to contribute to the advancement of this field by exploring various deep learning architectures and frameworks.


https://github.com/naayem/Medical-CT-Segmentation-DeepLearning/assets/14961905/fa89b6d8-f406-4e50-b8d1-e0f678b81abd


## Research Highlights

### State-of-the-Art Overview

- The thesis provides a comprehensive overview of the current state-of-the-art in medical image segmentation, highlighting the pivotal role of deep learning in this domain.

### Architecture Exploration

- Various deep learning architectures and frameworks were investigated, including U-Net and its variants, the nnUNet framework, and transformer-based models like UNETR and Swin UNETR.

### Performance Comparison

- Extensive experiments were conducted to compare the performance of these architectures, with a particular focus on assessing the state-of-the-art nnU-Net against the newer Swin-UNet model.

### Optimal Training Data

- The thesis aimed to approximate the optimal number of training images required to achieve the best segmentation results in the medical imaging domain.

### Future Directions

- The study discusses current trends and future directions in medical image segmentation, including the potential of transformer-based models, attention mechanisms, self-supervised learning, and domain adaptation.

## Repository Contents

This GitHub repository contains the following:

- Deep learning models used in the research.
- Evaluation methods and scripts for assessing model performance.
- Data preprocessing and augmentation tools.
- Documentation related to experiments and results.

## Usage

For detailed information on using the models and evaluation methods, please refer to the documentation provided in this repository.

## Conclusion

This thesis represents a significant contribution to the field of medical imaging, offering valuable insights into key advancements and identifying promising areas for future research and development. We encourage researchers and practitioners in the field to explore the models and methods made available in this repository.

## Contact

- Researcher: Vincent Naayem
- Supervisor: Prof. Alexandre Alahi
- University: EPFL (École polytechnique fédérale de Lausanne)

## Acknowledgments

Special thanks to all who contributed to this research, especially Prof. Alexandre Alahi for guidance and support.

## Galery
<div align=center>
   <img width="232" alt="Screenshot 2024-01-20 at 17 47 46" src="https://github.com/naayem/Medical-CT-Segmentation-DeepLearning/assets/14961905/7b8b0aa7-a60d-458b-9797-7e81d347812c">
   <img width="201" alt="Screenshot 2024-01-20 at 17 53 39" src="https://github.com/naayem/Medical-CT-Segmentation-DeepLearning/assets/14961905/1e40d79a-f918-4ed8-8c37-0cbf10806a2b">
</div>


# nnU-Net
[nnU-net](https://github.com/MIC-DKFZ/nnUNet)
nnU-Net makes the following contributions to the field:

1. **Standardized baseline:** nnU-Net is the first standardized deep learning benchmark in biomedical segmentation.
   Without manual effort, researchers can compare their algorithms against nnU-Net on an arbitrary number of datasets
   to provide meaningful evidence for proposed improvements.
2. **Out-of-the-box segmentation method:** nnU-Net is the first plug-and-play tool for state-of-the-art biomedical
   segmentation. Inexperienced users can use nnU-Net out of the box for their custom 3D segmentation problem without
   need for manual intervention.
3. **Framework:** nnU-Net is a framework for fast and effective development of segmentation methods. Due to its modular
   structure, new architectures and methods can easily be integrated into nnU-Net. Researchers can then benefit from its
   generic nature to roll out and evaluate their modifications on an arbitrary number of datasets in a
   standardized environment.

nnU-Net has been tested on Linux (Ubuntu 16, 18 and 20; centOS, RHEL). We do not provide support for other operating
systems.

nnU-Net requires a GPU! For inference, the GPU should have 4 GB of VRAM. For training nnU-Net models the GPU should have at
least 10 GB (popular non-datacenter options are the RTX 2080ti, RTX 3080 or RTX 3090). 

For training, it is recommended to have a strong CPU to go along with the GPU. At least 6 CPU cores (12 threads) are recommended. CPU
requirements are mostly related to data augmentation and scale with the number of input channels. They are thus higher
for datasets like BraTS which use 4 image modalities and lower for datasets like LiTS which only uses CT images.

It is very strongly recommended to install nnU-Net in a virtual environment.
[Here is a quick how-to for Ubuntu.](https://linoxide.com/linux-how-to/setup-python-virtual-environment-ubuntu/)
If you choose to compile pytorch from source, you will need to use conda instead of pip. In that case, please set the
environment variable OMP_NUM_THREADS=1 (preferably in your bashrc using `export OMP_NUM_THREADS=1`). This is important!

Python 2 is deprecated and not supported. Please make sure you are using Python 3.

1) Install [PyTorch](https://pytorch.org/get-started/locally/) as described on their website (conda/pip). Please 
install the latest version and (IMPORTANT!) choose 
the highest CUDA version compatible with your drivers for maximum performance. 
**DO NOT JUST `PIP INSTALL NNUNET` WITHOUT PROPERLY INSTALLING PYTORCH FIRST**
2) Verify that a recent version of pytorch was installed by running
    ```bash
    python -c 'import torch;print(torch.backends.cudnn.version())'
    python -c 'import torch;print(torch.__version__)'   
    ```
   This should print `8200` and `1.11.0+cu113` (Apr 1st 2022)
3) Install nnU-Net depending on your use case:
    1) For use as **standardized baseline**, **out-of-the-box segmentation algorithm** or for running **inference with pretrained models**:

       ```pip install nnunet```

    2) For use as integrative **framework** (this will create a copy of the nnU-Net code on your computer so that you can modify it as needed):
          ```bash
          git clone https://github.com/MIC-DKFZ/nnUNet.git
          cd nnUNet
          pip install -e .
          ```
4) nnU-Net needs to know where you intend to save raw data, preprocessed data and trained models. For this you need to
   set a few of environment variables. Please follow the instructions [here](documentation/setting_up_paths.md).
5) (OPTIONAL) Install [hiddenlayer](https://github.com/waleedka/hiddenlayer). hiddenlayer enables nnU-net to generate
   plots of the network topologies it generates (see [Model training](#model-training)). To install hiddenlayer,
   run the following commands:
    ```bash
    pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer
    ```

Installing nnU-Net will add several new commands to your terminal. These commands are used to run the entire nnU-Net
pipeline. You can execute them from any location on your system. All nnU-Net commands have the prefix `nnUNet_` for
easy identification.

Note that these commands simply execute python scripts. If you installed nnU-Net in a virtual environment, this
environment must be activated when executing the commands.

All nnU-Net commands have a `-h` option which gives information on how to use them.

A typical installation of nnU-Net can be completed in less than 5 minutes. If pytorch needs to be compiled from source
(which is what we currently recommend when using Turing GPUs), this can extend to more than an hour.

## Usage
To familiarize yourself with nnU-Net it is recommended you have a look at the (#Examples) of the github page of nnU-Net before you start with
your own dataset. All the documentation about the usage are available in the github pagel.

# swinUNETR
[swinUNETR](https://github.com/Project-MONAI/research-contributions)
This repository contains the code for Swin UNETR. Swin UNETR is the state-of-the-art on Medical Segmentation Decathlon (MSD) and Beyond the Cranial Vault (BTCV) Segmentation Challenge dataset. A novel methodology is devised for pre-training Swin UNETR backbone in a self-supervised manner. We provide the option for training Swin UNETR by fine-tuning from pre-trained self-supervised weights or from scratch.

This repository also contains the code for the task of brain tumor segmentation using the BraTS 21 challenge dataset. Swin UNETR ranked among top-perfoming models in BraTS 21 validation phase. 

A tutorial for BraTS21 brain tumor segmentation using Swin UNETR model is provided.

A tutorial for BTCV multi-organ segmentation using Swin UNETR model is provided.

This repository contains additionnaly the code for self-supervised pre-training of Swin UNETR model for medical image segmentation.

# Commands used to configure a scitas environment

To load the necessary modules of the clusters: (To do every session)
```
module load gcc python cuda
```

To create a virtual env: (Do it one time, To avoid problems with the existing libraries of the system the argument --system-site-packages is important)
```
virtualenv --system-site-packages venv-pytorch
```

To activate the environment: (each time you want to use the code)
```
source venv-pytorch/bin/activate
```

To install the compatible version of cuda:
```
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```

To install out of the box nnunet (see github page of nnunet):
```
pip install nnunet
```

Example of a help command of the out-of-the-box nnunet:
```
nnUNet_train -h
```
