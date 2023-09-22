from enum import Enum
from typing import Tuple, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray

# maps are defined as arrays
Map = NDArray[np.int_]

# location is a coordinate (row, column) in the grid
Location = Tuple[int, int]

maps: Dict[str, Map] = {
    "7x7": np.array(
        [
            [2, 0, 0, 0, 4, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 2],
            [0, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 0, 0, 1],
            [1, 1, 0, 3, 0, 1, 1],
        ],
        dtype=int,
    )
}


class Tile(Enum):
    """Defines different tile types."""

    EMPTY = 0
    WALL = 1
    GOAL = 2
    RUNNER_START = 3
    CHASER_START = 4


class Grid:
    def __init__(self, map_name: str) -> None:
        self._map = self._get_map(map_name)
        self._height = self._map.shape[0]
        self._width = self._map.shape[1]

    @staticmethod
    def _get_map(map_name: str) -> Map:
        """
        Returns the map (array) associated to the given name.
        """
        if map_name not in maps:
            raise Exception(
                f"Incorrect map name {map_name};",
                f"should be one of {list(maps.keys())}",
            )

        return maps[map_name]

    def is_valid_location(self, location: Location) -> bool:
        """
        Checks if the given location is valid, i.e. within the grid.
        """
        return (
            location[0] >= 0
            and location[0] < self._height
            and location[1] >= 0
            and location[1] < self._width
        )

    def is_empty(self, location: Location) -> bool:
        """
        Checks if the given location is valid and empty (not a wall).
        """
        # check if the location is valid
        if not self.is_valid_location(location):
            return False

        # check if location is not a wall
        is_wall = self._map[*location] == Tile.WALL.value

        return not is_wall

    def is_runner_goal(self, location: Location) -> bool:
        """
        Checks if the given location is the goal of the runner.
        """
        # check if the location is valid
        if not self.is_valid_location(location):
            return False

        return self._map[*location] == Tile.GOAL.value

    def adjacent_tiles(self, location: Location) -> List[Location]:
        """
        Returns a list of tiles that are immediately adjacent to the given location.
        There are between two and four such tiles; tiles outside the boundaries of
        the grid do not count.

        Keyword arguments:
        location: the location around which the tiles of interest lie.

        Returns:
        A list of locations of tiles adjacent to the given location.
        """
        row, col = location
        return [
            loc
            for loc in ((row + 1, col), (row - 1, col), (row, col - 1), (row, col + 1))
            if self.is_valid_location(loc)
        ]

    @property
    def runner_start_location(self) -> Location:
        """
        Returns the unique location in the grid where the runner starts.
        """
        runner_start_tiles = self._map == Tile.RUNNER_START.value
        runner_start_tiles = list(zip(*runner_start_tiles.nonzero()))

        # check that there is only one start tile
        assert len(runner_start_tiles) == 1

        return runner_start_tiles[0]

    @property
    def chaser_start_location(self) -> Location:
        """
        Returns the unique location in the grid where the chaser starts.
        """
        chaser_start_tiles = self._map == Tile.CHASER_START.value
        chaser_start_tiles = list(zip(*chaser_start_tiles.nonzero()))

        # check that there is only one start tile
        assert len(chaser_start_tiles) == 1

        return chaser_start_tiles[0]

    def render_grid(
        self, runner_location: Optional[Location], chaser_location: Optional[Location]
    ):
        """
        Displays the grid with walls replaced shown as # and goal locations shown as
        O. If runner_location or chaser_location are provided, the runner and chaser
        are shown by R and C.
        """
        printout = ""

        for row in range(self._height):
            for col in range(self._width):
                pass
        # find the walls
        # walls = self._map == Tile.WALL.value
