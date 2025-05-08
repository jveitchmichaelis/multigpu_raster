import os
import rasterio
from fixtures import TEST_IMAGE_PATH, TEST_IMAGE_SIZE, setup_image


def test_image_generation(setup_image):
    assert os.path.exists(TEST_IMAGE_PATH), "Test image was not created"
    with rasterio.open(TEST_IMAGE_PATH) as src:
        assert src.width == TEST_IMAGE_SIZE
        assert src.height == TEST_IMAGE_SIZE
        assert src.count == 3
