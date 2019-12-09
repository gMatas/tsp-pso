from typing import List

from tsppso.datatypes import FloatMatrix, Position


def euclidean_distance(a: Position, b: Position) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def distances_matrix(positions: List[Position]) -> FloatMatrix:
    n = len(positions)
    return tuple(
        tuple(
            0 if i == j else euclidean_distance(positions[i], positions[j])
            for j in range(n)
        ) for i in range(n))


def lshift(seq: list, k: int) -> list:
    n = len(seq)
    k = k - k // n * n if k > n else k
    right_seq = seq[k:]
    left_seq = seq[:k]
    shifted_seq = right_seq + left_seq
    return shifted_seq


def neighborhood_inversion(seq: List[int], i: int, j: int) -> List[int]:
    seq = seq.copy()
    if j > i:
        seq[i:j+1] = seq[i:j+1][::-1]
    elif i > j:
        n = len(seq)
        neighborhood = seq[i:] + seq[:j+1]
        inv_neighborhood = neighborhood[::-1]
        seq[i:] = inv_neighborhood[:n-i]
        seq[:j+1] = inv_neighborhood[n-i:]

    return seq
