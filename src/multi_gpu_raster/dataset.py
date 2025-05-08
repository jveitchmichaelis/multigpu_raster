import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from .tiling import Tiler


# Dataset for tiled reading
class TiledGeoTIFFDataset(Dataset):
    def __init__(self, image_path, tile_size):
        self.image_path = image_path
        self.tile_size = tile_size
        with rasterio.open(image_path) as src:
            self.width, self.height = src.width, src.height
            tiler = Tiler(
                self.width, self.height, tile_size=self.tile_size, min_overlap=256
            )
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
            return data
