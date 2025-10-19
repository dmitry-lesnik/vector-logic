"""
Helper functions for optimising StateVector multiplication.

This module contains functions for calculating and updating similarity matrices
based on the pivot sets of StateVectors. These matrices are used to guide the
multiplication strategy in the main engine, aiming to reduce the size of
intermediate results.
"""

from typing import List, Tuple

import numpy as np


def calc_ps_unions_intersections(pivot_sets: List[set[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the sizes of unions and intersections for a given list of pivot sets.

    Parameters
    ----------
    pivot_sets : list[set[int]]
        A list of sets, where each set contains 1-based integer indices of variables.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two NumPy arrays:
        - union_sizes: A square matrix where `union_sizes[i, j]`
        is the size of the union of `pivot_sets[i]` and `pivot_sets[j]`.
        - intersection_sizes: A square matrix where `intersection_sizes[i, j]`
        is the size of the intersection of `pivot_sets[i]` and `pivot_sets[j]`.
    """
    num_svs = len(pivot_sets)
    if num_svs == 0:
        return np.array([[]]), np.array([[]])
    max_idx = max([max(p_set) for p_set in pivot_sets])

    # 1. Create a boolean matrix where rows are pivot sets and columns are variables
    presence_matrix = np.zeros((num_svs, max_idx), dtype=bool)
    for i, p_set in enumerate(pivot_sets):
        if p_set:
            # Variable indices are 1-based, so we subtract 1 for 0-based NumPy indexing
            indices = np.array(list(p_set)) - 1
            presence_matrix[i, indices] = True

    # 2. Calculate intersection sizes using matrix multiplication
    # The dot product of the boolean matrix with its transpose gives the intersection sizes.
    intersection_sizes = presence_matrix.astype(np.int32) @ presence_matrix.astype(np.int32).T

    # 3. Calculate union sizes using the inclusion-exclusion principle
    # |A U B| = |A| + |B| - |A intersect B|
    set_lengths = np.sum(presence_matrix, axis=1, dtype=np.int32)
    # Use broadcasting to create the |A| + |B| matrix
    sum_of_lengths = set_lengths[:, np.newaxis] + set_lengths
    union_sizes = sum_of_lengths - intersection_sizes
    return union_sizes, intersection_sizes


def update_ps_unions_intersections(
    union_sizes: np.ndarray,
    intersection_sizes: np.ndarray,
    indices_to_remove: list[int],
    pivot_sets: List[set[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Efficiently update union and intersection size matrices.

    This function is called after a cluster of StateVectors has been multiplied.
    It removes the rows/columns corresponding to the original vectors and appends
    a new row/column for the resulting product vector.

    Parameters
    ----------
    union_sizes : np.ndarray
        The current union sizes matrix.
    intersection_sizes : np.ndarray
        The current intersection sizes matrix.
    indices_to_remove : list[int]
        A list of row/column indices to remove from the matrices.
    pivot_sets : List[set[int]]
        The *new* list of pivot sets, including the one for the newly
        created product vector (expected to be at the end).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the updated union and intersection size matrices.
    """
    for i in indices_to_remove:
        union_sizes = np.delete(union_sizes, i, axis=0)
        union_sizes = np.delete(union_sizes, i, axis=1)
        intersection_sizes = np.delete(intersection_sizes, i, axis=0)
        intersection_sizes = np.delete(intersection_sizes, i, axis=1)

    new_row_col_size = len(pivot_sets)
    new_union_sizes = np.zeros((new_row_col_size, new_row_col_size), dtype=int)
    new_union_sizes[:-1, :-1] = union_sizes
    new_intersection_sizes = np.zeros((new_row_col_size, new_row_col_size), dtype=int)
    new_intersection_sizes[:-1, :-1] = intersection_sizes

    for k in range(new_row_col_size - 1):
        new_union_sizes[k, -1] = new_union_sizes[-1, k] = len(pivot_sets[k].union(pivot_sets[-1]))
        new_intersection_sizes[k, -1] = new_intersection_sizes[-1, k] = len(pivot_sets[k].intersection(pivot_sets[-1]))
        new_union_sizes[-1, -1] = new_intersection_sizes[-1, -1] = len(pivot_sets[-1])
    return new_union_sizes, new_intersection_sizes


def find_next_cluster(
    pivot_sets: List[set[int]],
    union_sizes: np.ndarray,
    intersection_sizes: np.ndarray,
    max_cluster_size: int = 2,
) -> List[int]:
    """
    Finds the best cluster of state vectors to multiply next.

    This heuristic identifies the pair with the highest Jaccard similarity
    between their pivot sets, promoting earlier and more effective reduction.

    Parameters
    ----------
    pivot_sets : List[set[int]]
        A list of pivot sets, where each set contains 1-based integer indices of variables.
    union_sizes : np.ndarray
        A square matrix where `union_sizes[i, j]` is the size of the union of
        `pivot_sets[i]` and `pivot_sets[j]`.
    intersection_sizes : np.ndarray
        A square matrix where `intersection_sizes[i, j]` is the size of the intersection of
        `pivot_sets[i]` and `pivot_sets[j]`.
    max_cluster_size : int, optional
        The maximum number of pivot sets to include in the cluster, by default 2.

    Returns
    -------
    List[int]
        A list of indices representing the pivot sets chosen for the next cluster.

    """
    if len(pivot_sets) <= max_cluster_size:
        return list(range(len(pivot_sets)))

    # Replace division by zero with 0
    with np.errstate(divide="ignore", invalid="ignore"):
        scores_table = np.nan_to_num(intersection_sizes / union_sizes)

    np.fill_diagonal(scores_table, 0)

    row_scores = np.max(scores_table**2, axis=1)
    best_row_index = np.argmax(row_scores)
    scores_in_best_row = scores_table[best_row_index, :]

    # Get the indices that would sort the scores in descending order
    sorted_indices = np.argsort(scores_in_best_row)[::-1]

    # Remove best_row_index from sorted_indices
    sorted_indices = sorted_indices[sorted_indices != best_row_index]

    top_indices = [int(best_row_index)]
    for idx in sorted_indices[:max_cluster_size]:
        top_indices.append(int(idx))
        if scores_in_best_row[idx] == 0 and len(top_indices) > 1:
            break
        if len(top_indices) == max_cluster_size:
            break

    return top_indices
