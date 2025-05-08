import logging
import math
from typing import Generator, List, Tuple

import numpy as np
import numpy.typing as npt
from shapely.geometry import box

logger = logging.getLogger(__name__)


def generate_tiles(height, width, tile_size) -> list:
    """
    Generate non-overlapping tile extents covering a source image.
    """

    n_tiles_x = int(math.ceil(width / tile_size))
    n_tiles_y = int(math.ceil(height / tile_size))

    tiles = []

    for tx in range(n_tiles_x):
        for ty in range(n_tiles_y):
            minx = tx * tile_size
            miny = ty * tile_size

            maxx = minx + tile_size
            maxy = miny + tile_size

            tile_box = box(
                minx,
                miny,
                min(maxx, width),
                min(maxy, height),
            )

            tiles.append(tile_box)

    return tiles


class Tiler:
    """
    Helper class to generate tiles over a 2D extent. Can optionally generate tiles with centre weighting,
    but by default returns equally spaced tiles with edges that align with the edges of the image. The
    tiler first determines the minimum number of tiles required to cover an extent and then distributes
    the tiles across it.

    Tiles can be larger than the input size, though in this case you should get a single tile that over-extends
    the array.

    This class returns tile extents and does not have any dependence on the source image or array, end users
    should use TiledImage or TiledGeoImage.
    """

    def __init__(
        self,
        width: int,
        height: int,
        tile_size: int,
        min_overlap: int,
        centered: bool = False,
        align_edges: bool = True,
    ):
        """

        Construct a tiler with the desired output spec (generally a set tile size with a minimum overlap). The
        returned tiles will have at least the minimum overlap; overlap is maximised subject to the number
        of tiles required to cover the image.

        Args:
            width (int): Image width
            height (int): Image height
            tile_size (int): Tile size
            min_overlap (int): Minimum tile overlap
            centered (bool): Distribute tile centres rather than align tile edges
            exact_overlap (bool): Enforce tiles to align with image bounds

        """
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.overlap = min_overlap
        self.centered = centered
        self.stride = tile_size - min_overlap
        self.align_edges = align_edges

        if self.overlap >= tile_size:
            raise ValueError("Overlap must be less than tile size.")

        if (self.width - tile_size) <= 0 and (self.height - tile_size) <= 0:
            self.overlap = 0

    @property
    def tiles(self) -> Generator:
        """
        Returns a generator of tiles (tuples of x, y slices)
        """
        return self._tiles()

    def _n_tiles(self, distance: int) -> int:
        """
        Returns the number of intervals required to cover a distance
        """
        if distance <= self.tile_size:
            return 1

        intervals = math.ceil(
            (distance - self.tile_size) / (self.tile_size - self.overlap)
        )

        return 1 + intervals

    @property
    def effective_overlap(self):
        pass

    def _tile_edges(
        self, extent: int, tile_size: int, stride: int, n_tiles: int
    ) -> List[int]:
        """
        Returns a list of tile edges in ascending axis order (e.g. left -> right)
        """

        if self.align_edges:
            return np.linspace(0, extent - tile_size, n_tiles).astype(int)
        else:
            edges = [int(stride * i) for i in range(n_tiles)]

            if self.centered:
                overlap = (edges[-1] + tile_size) - extent
                edges = [e - overlap // 2 for e in edges]

        return edges

    def _tiles(self) -> Generator:
        """
        Internal function for generating tiles. Proceeds roughly as follows:

        1. Determine what stride we need (tile_size - overlap). Stride is the distance
        between tile edges.
        2. Determine how many tiles we need to cover each axis, given a particular overlap
        3. Determine the boundaries of each tile
        4. Lazily generate the tile slices

        Returns:
            tiles: generator of tuple(slice, slice) in xy order
        """

        n_x_tiles = self._n_tiles(self.width)
        n_y_tiles = self._n_tiles(self.height)

        self.x_edges = self._tile_edges(
            self.width, self.tile_size, self.stride, n_x_tiles
        )
        self.y_edges = self._tile_edges(
            self.height, self.tile_size, self.stride, n_y_tiles
        )

        for y in self.y_edges:
            y_start = y
            y_end = y_start + self.tile_size

            for x in self.x_edges:
                x_start = x
                x_end = x_start + self.tile_size

                yield (slice(x_start, x_end, 1), slice(y_start, y_end, 1))
