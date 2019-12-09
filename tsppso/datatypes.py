from collections import namedtuple
from typing import Tuple


FloatMatrix = Tuple[Tuple[float, ...], ...]


class Position(namedtuple("Position", "x y")):
    @staticmethod
    def make(xy: Tuple[float, float]):
        return Position._make(xy)
