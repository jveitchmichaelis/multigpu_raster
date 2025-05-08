import pytest
from multi_gpu_raster.util import generate_test_image
import os

TEST_IMAGE_PATH = "./test_image.tif"
TILE_SIZE = 1024
BATCH_SIZE = 2
TEST_IMAGE_SIZE=4096

@pytest.fixture(scope="session")
def setup_image():
    # Generate a small synthetic image for testing
    if not os.path.exists(TEST_IMAGE_PATH):
        generate_test_image(TEST_IMAGE_PATH, size=TEST_IMAGE_SIZE, tile_size=TILE_SIZE)
    yield
    if os.path.exists(TEST_IMAGE_PATH):
        os.remove(TEST_IMAGE_PATH)