from dataclasses import dataclass
from functools import total_ordering
from random import Random
from typing import Union, Tuple, List, Iterable, Sequence

from tsppso.datatypes import FloatMatrix, Position
from tsppso.utilities import distances_matrix, lshift, neighborhood_inversion


class Problem(object):
    def __init__(
            self,
            x: Sequence[float] = None,
            y: Sequence[float] = None,
            xy: Iterable[Union[Position, Tuple[float, float]]] = None
    ):
        self.__positions: List[Position]
        self.__distances: FloatMatrix

        if x is not None or y is not None:
            assert len(x) == len(y), "x and y sequences must be of the same length."
            self.__positions = [Position(xcoord, ycoord) for xcoord, ycoord in zip(x, y)]
        elif xy is not None:
            self.__positions = [Position(xcoord, ycoord) for xcoord, ycoord in xy]
        else:
            raise ValueError('Either xy or x and y arguments must not be None.')

        # Pre-compute distances matrix between points.
        self.__distances = distances_matrix(self.__positions)

    @property
    def positions(self) -> List[Position]:
        return self.__positions

    @property
    def distances(self) -> FloatMatrix:
        return self.__distances


@dataclass
@total_ordering
class Solution(object):
    sequence: List[int]
    cost: float
    best_sequence: List[int]
    best_cost: float

    def __eq__(self, other):
        return self.cost == other.cost

    def __lt__(self, other):
        return self.cost < other.cost


def pso_minimize(
        n: int,
        poolsize: int,
        distances: FloatMatrix,
        p1: float = 0.9,
        p2: float = 0.05,
        p3: float = 0.05,
        max_no_improv: int = 3,
        rng: Random = None
) -> Solution:

    rng = Random() if rng is None else rng

    # Initialize pool of solutions.
    solutions = list()
    base_indices = list(range(len(distances)))
    for i in range(poolsize):
        solution_indices = base_indices.copy()
        rng.shuffle(solution_indices)
        solution_cost = evaluate_cost(solution_indices, distances)
        solution = create_solution(solution_indices, solution_cost)
        solutions.append(solution)

    global_solution_index = solutions.index(min(solutions))
    global_solution = copy_solution(solutions[global_solution_index])

    counter = 0

    while n > 0:
        print('i:', counter, 'cost:', global_solution.cost)
        counter += 1

        for i, solution in enumerate(solutions):
            # Define Solution particles movement.
            velocity = define_velocity([p1, p2, p3], rng)

            if velocity == 0:  # move independently on it's own.
                move_solution_independently(solution, distances, max_no_improv, rng)
            elif velocity == 1:  # move toward personal best position.
                move_solution_to_personal_best(solution, distances)
            else:  # move toward swarm best position.
                move_solution_to_swarm_best(solution, global_solution, distances)

            if solution.cost < solution.best_cost:
                # Update each particle's personal best solution.
                solution.best_sequence = solution.sequence
                solution.best_cost = solution.cost

        global_solution_index = solutions.index(min(solutions))
        copy_solution_to(solutions[global_solution_index], global_solution)

        p1 *= 0.95
        p2 *= 1.01
        p3 = 1 - (p1 + p2)
        n -= 1

    return global_solution


def move_solution_independently(solution: Solution, distances: FloatMatrix, max_no_improv: int, rng: Random):
    sequence, delta_cost = neighborhood_inversion_search(solution.sequence, distances, max_no_improv, rng)
    solution.sequence = sequence
    solution.cost += delta_cost


def move_solution_to_personal_best(solution: Solution, distances: FloatMatrix):
    sequence, cost = path_relinking_search(
        solution.sequence, solution.best_sequence, solution.best_cost, distances)
    solution.sequence = sequence
    solution.cost = cost


def move_solution_to_swarm_best(solution: Solution, swarm_solution: Solution, distances: FloatMatrix):
    sequence, cost = path_relinking_search(
        solution.sequence, swarm_solution.sequence, swarm_solution.cost, distances)
    solution.sequence = sequence
    solution.cost = cost


def define_velocity(probas: List[float], rng: Random = None) -> int:
    assert sum(probas) == 1.0, 'Sum of all probabilities must be equal to 1.'
    indices = list(range(len(probas)))
    chosen_velocities_ids = rng.choices(indices, probas, k=1)
    return chosen_velocities_ids[0]  # select and return the first velocity id in the pool.


def evaluate_cost(seq: List[int], distances: FloatMatrix) -> float:
    cost = 0
    n = len(seq)
    for i in range(1, n):
        cost += distances[seq[i - 1]][seq[i]]
    return cost + distances[seq[-1]][seq[0]]


def path_relinking_search(
        origin: List[int],
        target: List[int],
        target_cost: float,
        distances: FloatMatrix
) -> Tuple[List[int], float]:

    best_seq = target
    best_cost = target_cost

    target_value = target[0]
    target_index = origin.index(target_value)
    seq = lshift(origin, target_index)

    n = len(target)
    for i in range(1, n - 1):
        target_value = target[i]
        right_seq = seq[i:]
        target_index = right_seq.index(target_value)  # target element index that is used as shifting distance.
        seq[i:] = lshift(right_seq, target_index)

        cost = evaluate_cost(seq, distances)
        if cost < best_cost:
            best_seq = seq.copy()
            best_cost = cost

    return best_seq, best_cost


def neighborhood_inversion_search(
        seq: List[int],
        distances: FloatMatrix,
        max_no_improv: int,
        rng: Random = None
) -> Tuple[List[int], float]:
    rng = Random() if rng is None else rng

    best_delta_cost = 0
    best_i = 0
    best_j = 0

    n = len(seq)  # sequence size.
    m = 2  # neighborhood size.
    no_improv_count = 0  # number of iterations with no improvement for current neighborhood size.

    while n - m > 1:
        # Generate neighborhood range [i, j].
        i = rng.randint(0, n - 1)
        j = i + m - 1
        j = j if j < n else j - n

        ia = seq[i - 1]
        ib = seq[i]
        ja = seq[j]
        jb = seq[j + 1 if j + 1 < n else 0]

        cost0 = distances[ia][ib] + distances[ja][jb]
        cost1 = distances[ia][ja] + distances[ib][jb]
        delta_cost = cost1 - cost0

        if delta_cost < best_delta_cost:
            best_delta_cost = delta_cost
            best_i = i
            best_j = j
            m += 1
            no_improv_count = 0
        else:
            no_improv_count += 1
            if no_improv_count >= max_no_improv:
                m += 1
                no_improv_count = 0

    result_seq = neighborhood_inversion(seq, best_i, best_j)
    return result_seq, best_delta_cost


def create_solution(sequence: List[int], cost: float) -> Solution:
    return Solution(sequence, cost, sequence, cost)


def copy_solution(solution: Solution) -> Solution:
    return Solution(solution.sequence, solution.cost, solution.best_sequence, solution.best_cost)


def copy_solution_to(src: Solution, dst: Solution):
    dst.sequence = src.sequence
    dst.cost = src.cost
    dst.best_sequence = src.best_sequence
    dst.best_cost = src.best_cost
