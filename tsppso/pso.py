import abc
from dataclasses import dataclass
from random import Random

from tsppso.tsp import Problem, Solution, pso_minimize


class PSO(abc.ABC):
    @abc.abstractmethod
    def minimize(self, problem: Problem) -> Solution:
        pass


@dataclass
class TSPPSO(PSO):
    n: int
    poolsize: int
    p1: float = 0.9
    p2: float = 0.05
    p3: float = 0.05
    max_no_improv1: int = 3
    max_no_improv2: int = 3
    rng: Random = None

    def minimize(self, problem: Problem) -> Solution:
        return pso_minimize(
            self.n, self.poolsize, problem.distances, self.p1, self.p2,
            self.p3, self.max_no_improv1, self.max_no_improv2, self.rng)
