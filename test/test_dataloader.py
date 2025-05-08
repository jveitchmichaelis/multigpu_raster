# test_script.py
from torch.utils.data import DataLoader
from fixtures import TEST_IMAGE_PATH, TILE_SIZE, BATCH_SIZE, setup_image
from multi_gpu_raster.main import TiledGeoTIFFDataset

def test_dataloader(setup_image):
    dataset = TiledGeoTIFFDataset(TEST_IMAGE_PATH, TILE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)
    for batch in dataloader:
        assert batch.shape == (
            BATCH_SIZE,
            3,
            TILE_SIZE,
            TILE_SIZE,
        ), "Batch dimensions are incorrect"
        break


def test_dataloader_workers(setup_image):
    dataset = TiledGeoTIFFDataset(TEST_IMAGE_PATH, TILE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=8)
    for batch in dataloader:
        assert batch.shape == (
            BATCH_SIZE,
            3,
            TILE_SIZE,
            TILE_SIZE,
        ), "Batch dimensions are incorrect"
        break
