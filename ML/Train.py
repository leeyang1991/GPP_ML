
from lytools import *
T = Tools()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from __global__ import *
from pprint import pprint
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
import glob

import pickle
import argparse
from typing import Optional
from functools import partial
import pandas as pd
import rasterio
from flux_load_data_vInf3 import flux_dataset, flux_dataloader
from flux_regress import RegressionModel_flux

from PIL import Image
from flux_load_data_vInf3 import load_raster
from flux_load_data_vInf3 import preprocess_image
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from torcheval.metrics import R2Score
from sklearn.metrics import r2_score

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar,TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torchgeo.trainers import BaseTask


from terratorch.models import EncoderDecoderFactory
from terratorch.datasets import HLSBands
from terratorch.tasks import PixelwiseRegressionTask
from terratorch.models.pixel_wise_model import freeze_module
from terratorch.models.backbones.prithvi_mae import PrithviViT

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

result_root_this_script = join(results_root,'ML')


class prithvi_terratorch(nn.Module):

    def __init__(self, prithvi_weight, model_instance, input_size):
        super(prithvi_terratorch, self).__init__()

        # load checkpoint for Prithvi_global

        self.weights_path = prithvi_weight
        self.checkpoint = torch.load(self.weights_path)
        self.input_size = input_size

        self.prithvi_model = model_instance

        self.prithvi_model.load_state_dict(self.checkpoint, strict=False)

    def freeze_encoder(self):
        freeze_module(self.prithvi_model)

    def forward(self, x, temp, loc, mask):
        latent, _, ids_restore = self.prithvi_model.forward(x, temp, loc, mask)

        return latent


class Do_Train():

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Train', result_root_this_script, mode=2)
        self.log_dir = join(self.this_class_arr,'log')
        T.mkdir(self.log_dir,force=True)
        self.conf()
        self.model_path = join(this_root,'Model/Prithvi_EO_V2_300M_TL.pt')

        pass

    def run(self):
        self.train()
        pass

    def conf(self):
        self.device =global_device
        self.n_channel = 6
        self.embed_dim = 1024
        self.patch_size = [1,16,16]
        self.num_heads = 16
        self.mlp_ratio = 4
        self.decoder_depth = 8
        self.head_dropout = 0.2

        self.n_frame = 1
        self.n_iteration = 100
        self.checkpoint_dir = self.log_dir
        self.metrics_dir = self.log_dir
        self.plots_dir = self.log_dir
        self.train_batch_size = 16
        self.test_batch_size = 16
        self.optim_name = 'AdamW'
        self.learning_rate = 5e-5
        self.sch = 1.
        self.drp = self.sch
        self.drp_rate = 0.2
        self.bn = self.sch
        # class_weights=config["class_weights"]
        self.input_size = [6,50,50]
        self.year_to_test = 2018
        # print('TEST YEAR', year_to_test)
        # means = config["data"][f"means_for{year_to_test}test"]
        # stds = config["data"][f"stds_for{year_to_test}test"]
        # merra_means = config["data"][f"merra_means_for{year_to_test}test"]
        # merra_stds = config["data"][f"merra_stds_for{year_to_test}test"]
        # gpp_means = config["data"][f"gpp_means_for{year_to_test}test"]
        # gpp_stds = config["data"][f"gpp_stds_for{year_to_test}test"]
        # test_year = config["test_year"]
        norm = "z-score-std"


    def get_mse(self,pred, targ):
        criterion = nn.MSELoss()
        mse_loss = criterion(pred, targ)
        return mse_loss.item()

    def get_mae(self,pred, targ):
        criterion = nn.L1Loss()
        mae_loss = criterion(pred, targ)
        return mae_loss.item()

    def get_r_sq(self,pred, targ):
        metric = R2Score()
        metric.update(pred, targ)
        r2 = metric.compute()
        if isinstance(r2, torch.Tensor) and r2.numel() > 1:
            return r2[0].item()
        else:
            return r2.item()

    def train(self):
        outpng_dir = join(self.this_class_png,'performance')
        T.mkdir(outpng_dir,force=True)
        climate_variable_list = ['temperature_2m', 'temperature_2m_min', 'temperature_2m_max', 'total_precipitation_sum']
        bands_list = ['B2','B3','B4','B5','B6','B7']
        fpath = join(data_root,'Dataframe/HLS_chips.df')
        df = T.load_df(fpath)
        df = df[df['tif_path_exists']]
        df = df.dropna(how='any')
        # T.print_head_n(df)
        # exit()


        # Test
        test_df = df[df['year'] == self.year_to_test]
        test_chips = test_df['tif_path'].tolist()
        pprint(test_chips)
        exit()
        test_chips = [tif_path.replace('[PROJECT_ROOT]/',this_root) for tif_path in test_chips]
        climate_test = test_df[climate_variable_list].astype(float).values.tolist()
        test_target = test_df['GPP'].tolist()

        # Train
        train_df = df[df['year'] != self.year_to_test]
        train_chips = train_df['tif_path'].tolist()
        train_chips = [tif_path.replace('[PROJECT_ROOT]/',this_root) for tif_path in train_chips]
        climate_train = train_df[climate_variable_list].astype(float).values.tolist()
        train_target = train_df['GPP'].tolist()

        # Means and Stds for Test
        bands_means = test_df[bands_list].mean().tolist()
        bands_stds = test_df[bands_list].std().tolist()
        climate_means = test_df[climate_variable_list].mean().tolist()
        climate_std = test_df[climate_variable_list].std().tolist()
        # gpp_means = [test_df['GPP'].mean()]
        # gpp_stds = [test_df['GPP'].std()]
        gpp_means = np.array([test_df['GPP'].mean()])
        gpp_stds = np.array([test_df['GPP'].std()])

        bands_means = np.array(test_df[bands_list].mean().tolist())
        bands_stds = np.array(test_df[bands_list].std().tolist())
        climate_means = np.array(test_df[climate_variable_list].mean().tolist())
        climate_std = np.array(test_df[climate_variable_list].std().tolist())
        gpp_means = np.array([test_df['GPP'].mean()])
        gpp_stds = np.array([test_df['GPP'].std()])

        flux_dataset_train = flux_dataset(train_chips, bands_means, bands_stds, climate_train,
                                          climate_means, climate_std, gpp_means, gpp_stds, train_target)

        flux_dataset_test = flux_dataset(test_chips, bands_means, bands_stds, climate_test,
                                         climate_means, climate_std, gpp_means, gpp_stds, test_target)

        datamodule = flux_dataloader(flux_dataset_train, flux_dataset_test, self.train_batch_size, self.test_batch_size)
        datamodule_ = flux_dataloader(flux_dataset_train, flux_dataset_train, self.train_batch_size, self.test_batch_size)

        wt_file = self.model_path

        prithvi_instance = PrithviViT(
            patch_size=self.patch_size,
            num_frames=self.n_frame,
            in_chans=self.n_channel,
            embed_dim=self.embed_dim,
            decoder_depth=self.decoder_depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            head_dropout=self.head_dropout,
            backbone_input_size=[1, 50, 50],
            encoder_only=False,
            padding=True,
        )

        prithvi_model = prithvi_terratorch(wt_file, prithvi_instance, [1, 50, 50])
        # exit()
        prithvi_model.freeze_encoder()

        model_comb = RegressionModel_flux(prithvi_model)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model_comb.parameters(), lr=self.learning_rate, weight_decay=0.05)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5)

        task = PixelwiseRegressionTask(None, None, model=model_comb, loss="mse", optimizer="AdamW")

        checkpoint_callback = ModelCheckpoint(monitor=task.monitor, save_top_k=1, save_last=True)

        num_epochs = self.n_iteration
        experiment = "carbon_flux"
        # default_root_dir = os.path.join("tutorial_experiments", experiment)
        default_root_dir = join(self.log_dir,experiment)
        logger = TensorBoardLogger(save_dir=default_root_dir, name=experiment)
        trainer = Trainer(
            # precision="16-mixed",
            accelerator=self.device,
            callbacks=[
                # RichProgressBar(),
                TQDMProgressBar(),
                checkpoint_callback,
                LearningRateMonitor(logging_interval="epoch"),
            ],
            max_epochs=num_epochs,
            default_root_dir=default_root_dir,
            log_every_n_steps=1,
            check_val_every_n_epoch=200
        )

        results = trainer.predict(model=task, datamodule=datamodule, return_predictions=True)
        pred_test = [i[0] for i in results]
        targ_test = [j['mask'] for j in flux_dataset_test]
        pred_test = np.concatenate(pred_test, axis=0)
        r_norm = r2_score(targ_test, pred_test)  # takes true, pred --R2 test set normalized scale (same as unnorm)

        mean_gpp = gpp_means.reshape(-1, 1, 1)  # Mean across height and width, for each channel
        stds_gpp = gpp_stds.reshape(-1, 1, 1)

        pred_final_unnorm = pred_test * stds_gpp + mean_gpp
        targ_final_unnorm = targ_test * stds_gpp + mean_gpp
        pred_final_unnorm = pred_final_unnorm.flatten()[
            :, None]  # np.reshape(pred_final_unnorm,(pred_final_unnorm.shape[1],1))
        targ_final_unnorm = targ_final_unnorm.flatten()[
            :, None]  # np.reshape(targ_final_unnorm,(targ_final_unnorm.shape[2],1))
        r2_unnorm = r2_score(targ_final_unnorm, pred_final_unnorm)  # true,pred
        mse_unnorm = (targ_final_unnorm - pred_final_unnorm) ** 2
        mae_unnorm = (np.abs(targ_final_unnorm - pred_final_unnorm))
        rel_err_unnorm = (np.abs(
            targ_final_unnorm - pred_final_unnorm) / targ_final_unnorm)  # (obs-exp)/obs -- obs: true reading, exp: model pred

        test_stack = np.hstack(
            (pred_test, targ_test, pred_final_unnorm, targ_final_unnorm, mse_unnorm, mae_unnorm, rel_err_unnorm))
        np.savetxt(
            f"{self.metrics_dir}test_eval_MSELoss_ep{num_epochs}_lr{self.learning_rate}_{self.optim_name}_sc{self.sch}_yr{self.year_to_test}.csv",
            test_stack, fmt='%10.6f', delimiter=',', newline='\n', header='pred_n, tar_n, pred, tar, mse,mae, rel_err')

        # save r2 figure
        plt.scatter(targ_final_unnorm, pred_final_unnorm, alpha=0.6)
        plt.plot([min(targ_final_unnorm), max(targ_final_unnorm)], [min(targ_final_unnorm), max(targ_final_unnorm)],
                 color='red', lw=2, label='Perfect fit')
        plt.xlabel('True GPP', fontsize=14)
        plt.ylabel('Predicted GPP', fontsize=14)
        plt.grid(True)
        plt.title('Zeroshot\nR2: ' + str(r2_unnorm))
        outf1 = join(outpng_dir,f'zeroshot_r2_{self.year_to_test}.png')
        plt.savefig(outf1,dpi=600)
        plt.close()
        # plt.show()
        plt.figure()
        print("Starting training")
        # exit()

        trainer.fit(model=task, datamodule=datamodule)
        results = trainer.predict(model=task, datamodule=datamodule, return_predictions=True)
        results_train = trainer.predict(model=task, datamodule=datamodule_, return_predictions=True)

        # Testing Phase -- test set data in batches
        pred_test = [i[0] for i in results]
        targ_test = [j['mask'] for j in flux_dataset_test]
        pred_train = [i[0] for i in results_train]
        targ_train = [j['mask'] for j in flux_dataset_train]

        # Concatenate predictions across batches
        pred_test = np.concatenate(pred_test, axis=0)
        targ_test = np.concatenate(targ_test, axis=0)[:, None]

        r_norm = r2_score(targ_test, pred_test)  # takes true, pred --R2 test set normalized scale (same as unnorm)

        # gpp_means, std as np.array
        mean_gpp = gpp_means.reshape(-1, 1, 1)  # Mean across height and width, for each channel
        stds_gpp = gpp_stds.reshape(-1, 1, 1)

        pred_final_unnorm = pred_test * stds_gpp + mean_gpp
        targ_final_unnorm = targ_test * stds_gpp + mean_gpp
        pred_final_unnorm = pred_final_unnorm.flatten()[
            :, None]  # np.reshape(pred_final_unnorm,(pred_final_unnorm.shape[1],1))
        targ_final_unnorm = targ_final_unnorm.flatten()[
            :, None]  # np.reshape(targ_final_unnorm,(targ_final_unnorm.shape[2],1))
        r2_unnorm = r2_score(targ_final_unnorm, pred_final_unnorm)  # true,pred
        mse_unnorm = (targ_final_unnorm - pred_final_unnorm) ** 2
        mae_unnorm = (np.abs(targ_final_unnorm - pred_final_unnorm))
        rel_err_unnorm = (np.abs(
            targ_final_unnorm - pred_final_unnorm) / targ_final_unnorm)  # (obs-exp)/obs -- obs: true reading, exp: model pred

        test_stack = np.hstack(
            (pred_test, targ_test, pred_final_unnorm, targ_final_unnorm, mse_unnorm, mae_unnorm, rel_err_unnorm))
        np.savetxt(
            f"{self.metrics_dir}test_eval_MSELoss_ep{num_epochs}_lr{self.learning_rate}_{self.optim_name}_sc{self.sch}_yr{self.year_to_test}.csv",
            test_stack, fmt='%10.6f', delimiter=',', newline='\n', header='pred_n, tar_n, pred, tar, mse,mae, rel_err')

        # save r2 figure
        plt.scatter(targ_final_unnorm, pred_final_unnorm, alpha=0.6)
        plt.plot([min(targ_final_unnorm), max(targ_final_unnorm)], [min(targ_final_unnorm), max(targ_final_unnorm)],
                 color='red', lw=2, label='Perfect fit')
        plt.xlabel('True GPP', fontsize=14)
        plt.ylabel('Predicted GPP', fontsize=14)
        plt.grid(True)
        plt.title('Test set\nR2: ' + str(r2_unnorm))
        outf2 = join(outpng_dir,f'test_r2_{self.year_to_test}.png')
        plt.savefig(outf2,dpi=600)
        plt.close()
        # plt.show()
        plt.figure()

        # evaluate performance on training data
        pred_test_tr = [i[0] for i in results_train]
        targ_test_tr = [j['mask'] for j in flux_dataset_train]

        # Concatenate predictions across batches
        pred_test_tr = np.concatenate(pred_test_tr, axis=0)
        targ_test_tr = np.concatenate(targ_test_tr, axis=0)[:, None]

        # unnormalize and save pred, metrics on full training set
        pred_final_unnorm_tr = pred_test_tr * stds_gpp + mean_gpp
        targ_final_unnorm_tr = targ_test_tr * stds_gpp + mean_gpp

        pred_final_unnorm_tr = pred_final_unnorm_tr.flatten()[:,
        None]  # np.reshape(pred_final_unnorm_tr,(pred_final_unnorm_tr.shape[1],1))
        targ_final_unnorm_tr = targ_final_unnorm_tr.flatten()[:,
        None]  # np.reshape(targ_final_unnorm_tr,(targ_final_unnorm_tr.shape[2],1))
        r2_unnorm_tr = r2_score(targ_final_unnorm_tr, pred_final_unnorm_tr)  # true,pred
        print(r2_unnorm_tr)
        # mse_norm= (test_pred - test_target) ** 2
        mse_unnorm_tr = (targ_final_unnorm_tr - pred_final_unnorm_tr) ** 2
        mae_unnorm_tr = (np.abs(targ_final_unnorm_tr - pred_final_unnorm_tr))
        rel_err_unnorm_tr = (np.abs(targ_final_unnorm_tr - pred_final_unnorm_tr) / targ_final_unnorm_tr)
        train_stack = np.hstack(
            (pred_test_tr, targ_test_tr, pred_final_unnorm_tr, targ_final_unnorm_tr, mse_unnorm_tr, mae_unnorm_tr,
             rel_err_unnorm_tr))
        np.savetxt(
            f"{self.metrics_dir}train_eval_MSELoss_ep{num_epochs}_lr{self.learning_rate}_{self.optim_name}_sc{self.sch}_yr{self.year_to_test}.csv",
            test_stack, fmt='%10.6f', delimiter=',', newline='\n', header='pred_n, tar_n, pred, tar, mse,mae, rel_err')

        # save r2 image
        plt.scatter(targ_final_unnorm_tr, pred_final_unnorm_tr, alpha=0.6)
        plt.plot([min(targ_final_unnorm_tr), max(targ_final_unnorm_tr)],
                 [min(targ_final_unnorm_tr), max(targ_final_unnorm_tr)],
                 color='red', lw=2, label='Perfect fit')
        plt.xlabel('True GPP', fontsize=14)
        plt.ylabel('Predicted GPP', fontsize=14)
        plt.grid(True)
        plt.title('Train set\nR2: ' + str(r2_unnorm_tr))
        # plt.show()
        outf3 = join(outpng_dir,f'train_r2_{self.year_to_test}.png')
        plt.savefig(outf3,dpi=600)
        plt.close()


def main():
    Do_Train().run()

if __name__ == '__main__':
    main()