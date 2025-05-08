import os
import torch
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf
import hydra
from tiling import Tiler
from torchvision.transforms import ToTensor
import time
import logging

logging.basicConfig(level=logging.INFO)

# Utility to generate a large test image
def generate_test_image(file_path, size=10000, tile_size=1024):
    with rasterio.open(
        file_path, 'w',
        driver='GTiff',
        height=size,
        width=size,
        count=3,
        dtype='uint8',
        compress='deflate'
    ) as dst:
        for i in range(1, 4):
            data = torch.randint(0, 256, (size, size), dtype=torch.uint8).numpy()
            dst.write(data, i)
    logging.info(f"Generated test image at {file_path}")

# Dataset for tiled reading
class TiledGeoTIFFDataset(Dataset):
    def __init__(self, image_path, tile_size):
        self.image_path = image_path
        self.tile_size = tile_size
        with rasterio.open(image_path) as src:
            self.width, self.height = src.width, src.height
            tiler = Tiler(self.width, self.height, tile_size=self.tile_size, min_overlap=256)
            self.tiles = list(tiler.tiles)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        xmin, ymin = self.tiles[idx]
        with rasterio.open(self.image_path) as src:
            slice_x, slice_y = self.tiles[idx]
            window = Window.from_slices(rows=slice_y, cols=slice_x)
            data = src.read(window=window, boundless=True)
            data = torch.tensor(data, dtype=torch.float32) / 255.0
            return ToTensor()(data)

# PyTorch Lightning model
class ResNet50Prediction(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.eval()

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx):
        output = self(batch)
        preds = torch.argmax(output, dim=1)
        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

# DataModule for Lightning
class TiledDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.dataset = TiledGeoTIFFDataset(self.config.image_path, self.config.tile_size)

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.workers,
            pin_memory=True
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

