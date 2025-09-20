import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import os
from torchgeo.datasets import NonGeoDataset
from torchgeo.datamodules import NonGeoDataModule
from lightning import LightningModule
from torch.utils.data import DataLoader
from lightning import LightningDataModule
NO_DATA = -0.9999
NO_DATA_FLOAT = 0.0001
PERCENTILES = (0.1, 99.9)


def load_raster(path, if_img, crop=None):
    with rasterio.open(path) as src:
        img = src.read()

        # load  selected 6 bands for Sentinnel 2 (S2)
        if if_img == 1:
            bands = [0, 1, 2, 3, 4, 5]
            img = img[bands, :, :]

        img = np.where(img == NO_DATA, NO_DATA_FLOAT,
                       img)  # update our NO_DATA with -0.9999 -- chips are already scaled
        # print("img size",img.shape)

        if crop:
            img = img[:, -crop[0]:, -crop[1]:]
    # print('return from load ras')
    return img


def preprocess_image(image, means, stds):
    # normalize image
    means1 = means.reshape(-1, 1, 1)  # Mean across height and width, for each channel
    stds1 = stds.reshape(-1, 1, 1)  # Std deviation across height and width, for each channel
    normalized = image.copy()
    normalized = ((image - means1) / stds1)

    normalized = torch.from_numpy(normalized.reshape(1, normalized.shape[0], 1, *normalized.shape[-2:])).to(
        torch.float32)
    # print('return from norm')
    return normalized


# consistent for HLS -- modified to add merra, flux -- these have z-score processing as hls
class flux_dataset(NonGeoDataset):

    def __init__(self, path, means, stds, merras_data, merra_means, merra_stds, gpp_mean, gpp_std, target):
        self.data_dir = path
        self.means = means
        self.stds = stds
        self.merras = merras_data
        self.merra_means = merra_means
        self.merra_stds = merra_stds
        self.gpp_means = gpp_mean
        self.gpp_stds = gpp_std
        self.target = target

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, idx):
        image_path = self.data_dir[idx]
        if_image = 1
        image = load_raster(image_path, if_image, crop=(50, 50))
        # print('hls after load raster', image.shape)
        final_image = preprocess_image(image, self.means, self.stds)
        # print('hls after preprocess', final_image.shape)
        merra = self.merras[idx]
        merra_vars = torch.from_numpy(np.array(self.merras[idx]).reshape(4, 1, 1)).to(torch.float32)

        mean_merra = self.merra_means.reshape(-1, 1, 1)  # Mean across height and width, for each channel
        stds_merra = self.merra_stds.reshape(-1, 1, 1)  # Std deviation across height and width, for each channel
        merra_vars_norm = (merra_vars - mean_merra) / (stds_merra)

        final_image = final_image.squeeze(0)

        mean_gpp = self.gpp_means.reshape(-1, 1, 1)  # Mean across height and width, for each channel
        stds_gpp = self.gpp_stds.reshape(-1, 1, 1)  # Std deviation across height and width, for each channel
        # print('mean, std gpp', mean_gpp, stds_gpp)
        gpp_vars_norm = (self.target[idx] - mean_gpp) / (stds_gpp)
        gpp_vars_norm = torch.from_numpy(np.array(gpp_vars_norm).reshape(1))
        # print('gpp is', gpp.shape)

        output = {"image": final_image.to(torch.float), "pt1d": merra_vars_norm.to(torch.float),
                  "mask": gpp_vars_norm.to(torch.float), "filename": image_path}

        return output  # final_image, merra_vars_norm, gpp_vars_norm


class flux_dataloader(LightningDataModule):

    def __init__(self, dataset_train=None, dataset_test=None, train_batch_size=None, test_batch_size=None):
        super().__init__()
        self.flux_dataset_train = dataset_train
        self.flux_dataset_test = dataset_test
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    def setup(self, stage: str = None):
        pass

    def foo(self):
        pass

    def train_dataloader(self):
        data_loader_flux_train = DataLoader(self.flux_dataset_train, batch_size=self.train_batch_size,
                                            shuffle=True)
        return data_loader_flux_train

    def test_dataloader(self):
        data_loader_flux_test = DataLoader(self.flux_dataset_test, batch_size=self.test_batch_size,
                                           shuffle=True)
        return data_loader_flux_test

    def predict_dataloader(self):
        data_loader_flux_test = DataLoader(self.flux_dataset_test, batch_size=self.test_batch_size,
                                           shuffle=True)
        return data_loader_flux_test

