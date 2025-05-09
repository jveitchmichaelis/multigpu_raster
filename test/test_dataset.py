import time

import torch
from fixtures import TEST_IMAGE_PATH, TILE_SIZE, setup_image

from multi_gpu_raster.dataset import DummyImageDataset, TiledGeoTIFFDataset


def test_image_dataset_loading(setup_image):
    dataset = TiledGeoTIFFDataset(TEST_IMAGE_PATH, TILE_SIZE)
    assert len(dataset) > 0, "Dataset should not be empty"
    data = dataset[0]
    assert isinstance(data, torch.Tensor), "Data should be a torch tensor"
    assert data.shape == (3, TILE_SIZE, TILE_SIZE), "Tile dimensions are incorrect"


def test_dummy_dataset_loading(setup_image):
    delay = 0.5
    count = 1000
    dataset = DummyImageDataset(delay, TILE_SIZE, TILE_SIZE, count=count)
    assert len(dataset) == count, "Dataset should not be empty"

    tstart = time.time()
    data = dataset[0]
    telapsed = time.time() - tstart
    assert abs(telapsed - 0.5) < 0.1
    assert isinstance(data, torch.Tensor), "Data should be a torch tensor"
    assert data.shape == (3, TILE_SIZE, TILE_SIZE), "Tile dimensions are incorrect"
