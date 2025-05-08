# test_script.py
import os
import pytest
import torch
import rasterio
from script import TiledGeoTIFFDataset, ResNet50Prediction, generate_test_image
from torch.utils.data import DataLoader
from torchvision.models import resnet50

# Test configuration
TEST_IMAGE_PATH = "./test_image.tif"
TILE_SIZE = 256
BATCH_SIZE = 2

@pytest.fixture(scope="session")
def setup_image():
    # Generate a small synthetic image for testing
    if not os.path.exists(TEST_IMAGE_PATH):
        generate_test_image(TEST_IMAGE_PATH, size=512, tile_size=TILE_SIZE)
    yield
    if os.path.exists(TEST_IMAGE_PATH):
        os.remove(TEST_IMAGE_PATH)

def test_imports():
    try:
        import script
        import pytorch_lightning
        import hydra
        assert True
    except ImportError:
        pytest.fail("Required modules could not be imported")

def test_image_generation(setup_image):
    assert os.path.exists(TEST_IMAGE_PATH), "Test image was not created"
    with rasterio.open(TEST_IMAGE_PATH) as src:
        assert src.width == 512
        assert src.height == 512
        assert src.count == 3

def test_dataset_loading(setup_image):
    dataset = TiledGeoTIFFDataset(TEST_IMAGE_PATH, TILE_SIZE)
    assert len(dataset) > 0, "Dataset should not be empty"
    data = dataset[0]
    assert isinstance(data, torch.Tensor), "Data should be a torch tensor"
    assert data.shape == (3, TILE_SIZE, TILE_SIZE), "Tile dimensions are incorrect"

def test_model_inference():
    model = ResNet50Prediction()
    x = torch.randn((1, 3, 224, 224))
    output = model(x)
    assert output.shape[1] == 1000, "ResNet50 output dimension mismatch"

def test_dataloader(setup_image):
    dataset = TiledGeoTIFFDataset(TEST_IMAGE_PATH, TILE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)
    for batch in dataloader:
        assert batch.shape == (BATCH_SIZE, 3, TILE_SIZE, TILE_SIZE), "Batch dimensions are incorrect"
        break

