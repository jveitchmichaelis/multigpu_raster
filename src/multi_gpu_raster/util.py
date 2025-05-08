import numpy as np
import rasterio
from rasterio.windows import Window
import logging


def generate_test_image(file_path, size=20000, tile_size=1024):
    """Generate a large tiled test image."""

    profile = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 3,
        "dtype": "uint8",
        "compress": "deflate",
        "tiled": True,
        "blockxsize": tile_size,
        "blockysize": tile_size,
        "bigtiff": "IF_NEEDED"
    }

    with rasterio.open(file_path, "w", **profile) as dst:
        for band in range(1, profile["count"] + 1):
            logging.info(f"Writing band {band}")
            for y in range(0, size, tile_size):
                for x in range(0, size, tile_size):
                    width = min(tile_size, size - x)
                    height = min(tile_size, size - y)

                    data = np.random.randint(
                        0, 256, (height, width), dtype=np.uint8
                    )

                    window = Window(x, y, width, height)
                    dst.write(data, band, window=window)

    logging.info(f"Generated test image at {file_path}")
