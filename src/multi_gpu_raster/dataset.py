import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset
import time

from .tiling import Tiler


# Dataset for tiled reading
class TiledGeoTIFFDataset(Dataset):
    def __init__(self, image_path, tile_size):
        self.image_path = image_path
        self.tile_size = tile_size
        with rasterio.open(image_path) as src:
            self.width, self.height = src.width, src.height
            tiler = Tiler(
                self.width, self.height, tile_size=self.tile_size, min_overlap=0
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
        

class DummyImageDataset(Dataset):
    def __init__(self, delay=0.01, width=1024, height=1024, count=10000):
        self.delay = delay
        self.width = width
        self.height = height
        self.count = count

    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        time.sleep(self.delay)
        return torch.rand(3, self.width, self.height).float()
    