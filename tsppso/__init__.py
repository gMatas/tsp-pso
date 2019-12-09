from tsppso.pso import TSPPSO
from tsppso.tsp import Problem
from tsppso.datatypes import Position


def canada():
    from tsppso.datasets import TSPDataset
    dataset = TSPDataset()
    dataset.read('../datasets/canada.tsp')

    import random
    rng = random.Random(97855)

    problem = Problem(xy=dataset.positions)
    optimizer = TSPPSO(200, 20, max_no_improv1=10, rng=rng)
    solution = optimizer.minimize(problem)

    print()
    print('seq:\n{}\n\ncost:\n{}'.format(solution.sequence, solution.cost))


def test():
    def brute_force_tsp(problem: Problem):
        import sys
        from itertools import permutations
        from tsppso.tsp import evaluate_cost
        min_seq = None
        min_cost = sys.float_info.max
        base_seq = list(range(len(problem.distances)))
        for seq in permutations(base_seq):
            cost = evaluate_cost(seq, problem.distances)
            if cost <= min_cost:
                min_cost = cost
                min_seq = seq
        return min_seq, min_cost

    positions = [
        (2, 9),  # A
        (8, 9),  # B
        (3, 7),  # C
        (1, 6),  # D
        (7, 6),  # E
        (4, 4),  # F
        (7, 1),  # G
    ]

    problem = Problem(xy=positions)

    seq, cost = brute_force_tsp(problem)
    print("seq:", list(seq), "cost:", cost)

    import random
    seed = random.randint(0, 100000)
    print("seed:", seed)

    optimizer = TSPPSO(100, 30, max_no_improv1=10, rng=random.Random(seed))
    solution = optimizer.minimize(problem)


if __name__ == "__main__":
    # test()
    canada()
