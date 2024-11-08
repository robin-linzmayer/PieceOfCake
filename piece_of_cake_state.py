from dataclasses import dataclass

from shapely import Polygon


@dataclass
class PieceOfCakeState:
    polygons: list[Polygon]
    cur_pos: tuple[int, int]
    turn_number: int
    requests: list[float]
    cake_len: float
    cake_width: float
    time_remaining: float
