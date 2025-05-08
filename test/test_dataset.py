import torch
from fixtures import TEST_IMAGE_PATH, TILE_SIZE, setup_image
from multi_gpu_raster.dataset import TiledGeoTIFFDataset

def test_dataset_loading(setup_image):
    dataset = TiledGeoTIFFDataset(TEST_IMAGE_PATH, TILE_SIZE)
    assert len(dataset) > 0, "Dataset should not be empty"
    data = dataset[0]
    assert isinstance(data, torch.Tensor), "Data should be a torch tensor"
    assert data.shape == (3, TILE_SIZE, TILE_SIZE), "Tile dimensions are incorrect"
