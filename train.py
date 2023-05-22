import sys
sys.path.insert(0,'./keops')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["USE_KEOPS"] = "True";

import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader
from datasets import PlanetoidDataset, TadpoleDataset
import pytorch_lightning as pl
from DGMlib.model_dDGM import DGM_Model
from pytorch_lightning import Callback

from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger     

def run_training_process(run_params):
    
    train_data = None
    test_data = None
    
    if run_params.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        train_data = PlanetoidDataset(split='train', name=run_params.dataset, device='cuda')
        val_data = PlanetoidDataset(split='val', name=run_params.dataset, samples_per_epoch=1)
        test_data = PlanetoidDataset(split='test', name=run_params.dataset, samples_per_epoch=1)
        
    if run_params.dataset == 'tadpole':
        train_data = TadpoleDataset(fold=run_params.fold,train=True, device='cuda')
        val_data = test_data = TadpoleDataset(fold=run_params.fold, train=False,samples_per_epoch=1)
                                   
    if train_data is None:
        raise Exception("Dataset %s not supported" % run_params.dataset)
        
    train_loader = DataLoader(train_data, batch_size=1,num_workers=0)
    val_loader = DataLoader(val_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=1)

    class MetricsCallback(Callback):
        def __init__(self):
            super().__init__()
            self.metrics = []

        def on_test_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            save_path = os.path.join(os.getcwd(), "stats")

            dgm_layers = run_params.dgm_layers
            # check if it's empty, for case #dgmlayers=0
            if dgm_layers[0]:
                emb_dims=dgm_layers[0][2]
            else:
                emb_dims=0

            # get #dgm_layers, exclude empty elements
            dgm_num = sum(1 for x in dgm_layers if x)

            # embedding function f
            f = run_params.ffun 

            # diffusion function g
            g = run_params.gfun

            paths = {
                "dataset": run_params.dataset,
                "k": run_params.k,
                "dist": run_params.distance,
                "dims": emb_dims,
                "dgm": dgm_num,
                "f": f,
                "g": g
            }

            with open(os.path.join(save_path,f'{paths["dataset"]}_test_acc_'\
                                f'k_{paths["k"]}_{paths["dist"]}_'\
                                f'd_{paths["dims"]}_'\
                                f'dgm_{paths["dgm"]}_' \
                                f'f_{paths["f"]}_' \
                                f'g_{paths["g"]}.txt'.lower()), "a+") as f:
                f.write(str(metrics["test_acc"].item()))
                f.write("\n")
            
    class MyDataModule(pl.LightningDataModule):
        def setup(self,stage=None):
            pass
        def train_dataloader(self):
            return train_loader
        def val_dataloader(self):
            return val_loader
        def test_dataloader(self):
            return test_loader
    
    
    #configure input feature size
    if run_params.pre_fc is None or len(run_params.pre_fc)==0: 
        if len(run_params.dgm_layers[0])>0:
            run_params.dgm_layers[0][0]=train_data.n_features
        run_params.conv_layers[0][0]=train_data.n_features
    else:
        run_params.pre_fc[0]=train_data.n_features
    run_params.fc_layers[-1] = train_data.num_classes
    
    model = DGM_Model(run_params)

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=20,
        verbose=False,
        mode='min')

    metrics_callback = MetricsCallback()
    callbacks = [checkpoint_callback,early_stop_callback,metrics_callback]
    
    if val_data==test_data:
        callbacks = None
        
    logger = TensorBoardLogger("logs/")
    trainer = pl.Trainer.from_argparse_args(run_params,logger=logger,
                                            callbacks=callbacks)
    
    trainer.fit(model, datamodule=MyDataModule())
    trainer.test()
    
if __name__ == "__main__":

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args(['--gpus','1',                         
                              '--log_every_n_steps','100',                          
                              '--max_epochs','100',
                              '--progress_bar_refresh_rate','10',                         
                              '--check_val_every_n_epoch','1'])
    parser.add_argument("--num_gpus", default=10, type=int)
    
    parser.add_argument("--dataset", default='Cora')
    parser.add_argument("--fold", default='0', type=int) #Used for k-fold cross validation in tadpole/ukbb
    
    
    parser.add_argument("--conv_layers", default=[[32,32],[32,16],[16,8]], type=lambda x :eval(x))
    parser.add_argument("--dgm_layers", default= [[32,16,4],[],[]], type=lambda x :eval(x))
    parser.add_argument("--fc_layers", default=[8,8,3], type=lambda x :eval(x))
    parser.add_argument("--pre_fc", default=[-1,32], type=lambda x :eval(x))

    parser.add_argument("--gfun", default='gcn')
    parser.add_argument("--ffun", default='gcn')
    parser.add_argument("--k", default=5, type=int) 
    parser.add_argument("--pooling", default='add')
    parser.add_argument("--distance", default='euclidean')

    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--test_eval", default=10, type=int)

    parser.set_defaults(default_root_path='./log')
    params = parser.parse_args(namespace=params)

    run_training_process(params)
