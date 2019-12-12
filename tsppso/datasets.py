from typing import List

from tsppso.datatypes import Position


class TSPDataset(object):
    def __init__(self, positions: List[Position] = None):
        self.positions: List[Position] = list() if positions is None else positions

    def read(self, filepath: str):
        with open(filepath, 'r') as fs:
            is_node_coord_section = False
            for line in fs:
                if not is_node_coord_section:
                    is_node_coord_section = line.startswith('NODE_COORD_SECTION')
                    continue
                elif line.startswith('EOF'):
                    break
                lineparts = line.split(' ')
                x = float(lineparts[1])
                y = float(lineparts[2])
                position = Position(x, y)
                self.positions.append(position)

    def unique(self, eps: float = 1e-6, inplace: bool = False) -> List[Position]:
        sorted_positions = sorted(self.positions, key=(lambda pos: pos.x if pos.x <= pos.y else pos.y))
        unique = list()
        pos_a = sorted_positions[0]
        n = len(sorted_positions)
        for i in range(1, n - 1):
            pos_b = sorted_positions[i]
            if abs(pos_a.x - pos_b.x) + abs(pos_a.y - pos_b.y) > eps:
                unique.append(pos_a)
            pos_a = sorted_positions[i]

        pos_b = sorted_positions[-1]
        if abs(pos_a.x - pos_b.x) + abs(pos_a.y - pos_b.y) > eps:
            unique.append(pos_a)
            unique.append(pos_b)
        else:
            unique.append(pos_a)

        if inplace:
            self.positions = unique

        return unique
