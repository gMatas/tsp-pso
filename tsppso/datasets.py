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
