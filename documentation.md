# Current Issues 
1. Experiments 5.1 (5.1.1, 5.1.2, 5.1.3, 5.1.4, 5.1.5) for discrete DGM are not consistent. 
2. Point Cloud 3D experiments are based on DGCNN: https://github.com/WangYueFt/dgcnn
    - Successfully re-run experiments in dgcnn.
    - Section 5.3 describes that kNN sampling scheme by DGCNN was replaced by the discrete sampling strategy of DGM. This experiment is demonstrated only for the **discrete** case. 

3. Zero-shot learning application is also done with **discrete** DGM. Not much information was provided to reproduce experiments in this section.

**Currently asking two co-authors on this issue, no update yet**
Luca Cosmo: luca.cosmo@unive.it
Kazi Anees: akazi1@mgh.harvard.edu

# Code Info
## Dataset
- Synthetic datasets for testing discrete DGM: Citeseer, PubMed, CiteSeer
- Classification for disease, ages: Tadpole, UK Biobank
    - Currently only *transductive setting* is provided.
- Computer Vision 3D application on point clouds: ShapeNet (only discrete DGM).

## Available settings
- Multiple configurations for discrete case.
- Citeseer, Pubmed, Citeseer epxeriments.

## Not (completely) available
- Tadpole, UK Biobank requires some modifications for the model from the descriptions of the paper.
- **Continuous** layer is provided, but the continuous model (i.e. definitions, training, etc) is not provided.
- Continuous model for tadpole, UK Biobank also requires modifications.

## Important files
### ./DGM_pytorch/DGMlib
1. model_dDGM.py: discrete DGM model definition, training/validation code
2. layers.py: include some different layers for experiments
    - Euclidean distance
    - Poincare distance
    - Discrete DGM module
    - Continuous DGM module
    - MLP module
    - Identity module
### ./DGM_pytorch/train.py
Main file to run experiments: simply run **python train.py** with parameters:

1. --num_gpus: total number of gpus
2. --dataset: which dataset used to train (UKBiobank, Tadpole require additional changes)
3. --fold: k-fold validation (for UKBiobank, Tadpole)
4. --conv_layers: number of convolutional layers
5. --dgm_layers: number of dgm layers
6. --fc_layers: number of linear layers
7. --pre_lc: pre linear layer setting
8. --gfun: diffusion function types: use state-of-the-art layers: gcn, gat, edgeconv
9. --ffun: graph embedding function types: use state-of-the-art layers: gcn, gat (+mlp, id for experiments)
10. --k: k param for k-gumbel sampling
11. --pooling: pooling type (default = add)
12. --dropout: drop out probability during training (not touching)
13. --lr: learning rate (not touching)
14. --test_eval: number of epoch for evaluation (not touching)

## Notes: all settings above are for discrete case sampling.

