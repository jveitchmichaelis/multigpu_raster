import logging
import os
import time

import hydra
import pytorch_lightning as pl
import rasterio
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from .dataset import TiledGeoTIFFDataset
from .model import ResNet50Prediction
from .util import generate_test_image

logging.basicConfig(level=logging.INFO)


# DataModule for Lightning
class TiledDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.dataset = TiledGeoTIFFDataset(
            self.config.image_path, self.config.tile_size
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.workers,
            pin_memory=True,
        )


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logging.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Generate a test image if it doesn't exist
    if not os.path.exists(cfg.image_path):
        generate_test_image(cfg.image_path)

    # Data loading
    dm = TiledDataModule(cfg)

    # Model and trainer setup
    model = ResNet50Prediction()
    trainer = pl.Trainer(
        devices=cfg.gpus,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=cfg.precision,
        profiler=cfg.profiler,
        log_every_n_steps=cfg.log_interval,
    )

    start_time = time.time()
    trainer.predict(model, datamodule=dm)
    end_time = time.time()

    os.remove(cfg.image_path)

    logging.info(f"Prediction completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
