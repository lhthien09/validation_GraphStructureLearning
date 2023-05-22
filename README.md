# DGM_pytorch

Code for the paper "Differentiable Graph Module (DGM) for Graph Convolutional Networks" by Anees Kazi*, Luca Cosmo*, Seyed-Ahmad Ahmadi, Nassir Navab, and Michael Bronstein


## Installation

Create a Conda virtual environment and install all the necessary packages

```
conda create -n DGMenv python=3.8
conda activate DGMenv
```

```
conda install -c anaconda cmake=3.19
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.1 -c pytorch
pip install pytorch_lightning==1.3.8

pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.8.1+cu101.html
pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.8.1+cu101.html
pip install torch-geometric
```

## Training

To train a model with the default options run the following command:
```
python train.py
``` 

## Notes
The graph sampling code is based on a modified version of the KeOps libray (www.kernel-operations.io) to speed-up the computation. In particular, the argKmin function of the original libray has been modified to handle the stochasticity of the sampling strategy, adding samples drawn from a Gumbel distribution to the input before performing the reduction.
