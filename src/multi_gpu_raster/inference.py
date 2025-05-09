import logging
import time

import hydra
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .dataset import DummyImageDataset, TiledGeoTIFFDataset
from .model import ObjectDetector
from .util import generate_test_image

logging.basicConfig(level=logging.INFO)


# DataModule for Lightning
class TiledDataModule(pl.LightningDataModule):
    def __init__(self, config, fake_data=True):
        super().__init__()
        self.config = config
        self.fake_data = fake_data

    def setup(self, stage=None):
        if self.fake_data:
            self.dataset = DummyImageDataset(
                self.config.dummy_dataset_delay,
                self.config.tile_size,
                self.config.tile_size,
                count=self.config.dummy_dataset_count,
            )
        else:
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


@hydra.main(config_path="conf", config_name="config", version_base=None)
def test_image(cfg: DictConfig):
    generate_test_image(cfg.image_path, cfg.image_size)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    logging.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Model and trainer setup
    model = ObjectDetector()

    trainer = pl.Trainer(
        devices=cfg.gpus if cfg.accelerator not in ["cpu", "mps"] else 1,
        accelerator=cfg.accelerator,
        strategy=(
            DDPStrategy(find_unused_parameters=False)
            if cfg.accelerator not in ["cpu", "mps"]
            else "auto"
        ),
        precision=cfg.precision if cfg.accelerator != "cpu" else "bf16-mixed",
        profiler=cfg.profiler,
        log_every_n_steps=cfg.log_interval,
        logger=CSVLogger("logs"),
    )

    # Data loading
    dm = TiledDataModule(cfg)

    start_time = time.time()
    trainer.predict(model, datamodule=dm)
    end_time = time.time()

    logging.info(f"Prediction completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
