# Empirical Validations of Graph Structure Learning methods for Citation Network Applications

Code for the BSc Thesis "Empirical Validations of Graph Structure Learning methods for Citation Network Applications" by Hoang Thien Ly at Faculty of Mathematics and Information Science, Warsaw University of Technology.

## Abstract

This Bachelor’s Thesis aims to examine the classification accuracy of graph structure learning methods in graph neural networks domain, with a focus on classifying a paper
in citation network datasets. Graph neural networks (GNNs) have recently emerged as a powerful machine learning concept allowing to generalize successful deep neural archi-
tectures to non-Euclidean structured data with high performance. However, one of the limitations of the majority of current GNNs is the assumption that the underlying graph
is known and fixed. In practice, real-world graphs are often noisy and incomplete or might even be completely unknown. In such cases, it would be helpful to infer the graph
structure directly from the data. Additionally, graph structure learning permits learning latent structures, which may increase the understanding of GNN models by providing edge
weights among entities in the graph structure, allowing modellers to further analysis.

As part of the work, we will:
* review the current state-of-the-art graph structure learning (GSL) methods.
* empirically validate GSL methods by accuracy scores with citation network datasets.
* analyze the mechanism of these approaches and analyze the influence of hyperparameters on model’s behavior.
* discuss future work.
Keywords: graph neural network, graph structure learning, empirical validations, citation network applications.

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

For the dDGM framework, to train a model with the default options run the following command:
```
python train.py
``` 

Other frameworks, codes are presented in the Jupyter notebook.
