import logging

import numpy as np
import rasterio


# Utility to generate a large test image
def generate_test_image(file_path, size=20000):
    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=3,
        dtype="uint8",
        compress="deflate",
    ) as dst:
        for i in range(1, 4):
            data = np.random.randint(0, 255, (size, size), dtype=np.uint8)
            dst.write(data, i)
    logging.info(f"Generated test image at {file_path}")
