"""
Helper functions for optimising StateVector multiplication.

This module contains functions for calculating and updating similarity matrices
based on the pivot sets of StateVectors. These matrices are used to guide the
multiplication strategy in the main engine, aiming to reduce the size of
intermediate results.
"""

from typing import List, Tuple, Optional

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

    if num_svs == 1:
        return np.array([[len(pivot_sets[0])]]), np.array([[len(pivot_sets[0])]])

    max_idx_list = [max(p_set) for p_set in pivot_sets if p_set]
    max_idx = max(max_idx_list) if max_idx_list else 0
    if max_idx == 0:
        # Handle all empty pivot sets
        # notice that variable indices are 1-based, hence max_idx=0 is only possible if all pivot sets are empty
        return np.zeros((num_svs, num_svs), dtype=int), np.zeros((num_svs, num_svs), dtype=int)

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
    sorted_indices = sorted(indices_to_remove, reverse=True)
    for i in sorted_indices:
        union_sizes = np.delete(union_sizes, i, axis=0)
        union_sizes = np.delete(union_sizes, i, axis=1)
        intersection_sizes = np.delete(intersection_sizes, i, axis=0)
        intersection_sizes = np.delete(intersection_sizes, i, axis=1)

    N = len(union_sizes)  # size before taking into account newly appended pivot sets
    N1 = len(pivot_sets)  # new size
    new_union_sizes = np.zeros((N1, N1), dtype=int)
    new_union_sizes[:N, :N] = union_sizes
    new_intersection_sizes = np.zeros((N1, N1), dtype=int)
    new_intersection_sizes[:N, :N] = intersection_sizes

    for i in range(N, N1):
        for k in range(i):
            new_union_sizes[i, k] = new_union_sizes[k, i] = len(pivot_sets[k].union(pivot_sets[i]))
            new_intersection_sizes[i, k] = new_intersection_sizes[k, i] = len(pivot_sets[k].intersection(pivot_sets[i]))
        new_union_sizes[i, i] = new_intersection_sizes[i, i] = len(pivot_sets[i])
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
    for idx in sorted_indices[: max_cluster_size - 1]:  # Corrected loop limit
        if scores_in_best_row[idx] == 0 and len(top_indices) > 1:
            break
        top_indices.append(int(idx))

    return top_indices


def find_predator_prey(
    sv_sizes: List[int],
    intersection_sizes: np.ndarray,
    base: float = 0.7,
    threshold: float = 1.5,
    max_predator_size: int = 2,
) -> Tuple[Optional[int], Optional[List[int]]]:
    """
    Finds one "predator" state vector and a list of "prey" state vectors.

    The "prey" state vectors will be multiplied by the "predator". The idea
    is that the expected size of the "prey" state vectors should shrink.

    We estimate the size of the product of two state vectors to be
    `n1 * n2 * base^m`, where `n1` and `n2` are the sizes of the operands,
    and `m` is the size of the intersection of their pivot sets.

    The relative reduction score of the second operand is:
    `score = size_before / size_after = 1 / (n1 * base^m)`
    If `score > 1`, the size of vector `n2` is expected to shrink.

    This method calculates a matrix of these scores and looks for a "predator"
    row `i` (where `n_i` is small, constrained by `max_predator_size`) that
    has high scores (e.g., > `threshold`) against multiple "prey" columns.

    The method finds the best row, in which we have more than one score
    that is bigger than 1. For every row, we calculate a row-score - the
    mean of squares of those scores that are bigger than `threshold`.

    If there is any row-score greater than 0, the method returns the
    index of the best row as a "predator" index, and a list of indices of
    those columns where the score is bigger than 1 as "prey" indices.

    Otherwise, the method returns (None, None).

    Parameters
    ----------
    sv_sizes : List[int]
        A list of the sizes (number of TObjects) of the StateVectors.
    intersection_sizes : np.ndarray
        A square matrix of pivot set intersection sizes.
    base : float, optional
        The base for the exponential reduction estimation. Defaults to 0.8.
    threshold : float, optional
        The minimum row-score to trigger the predator-prey optimisation. Defaults to 1.5.
    max_predator_size: int
        The maximum size (number of TObjects) a StateVector can have to be considered a predator.

    Returns
    -------
    Tuple[Optional[int], Optional[List[int]]]
        - (predator_index, [prey_index_1, prey_index_2, ...]) if a predator is found.
        - (None, None) otherwise.
    """
    num_svs = len(sv_sizes)
    if num_svs < 3:
        return None, None

    if min(sv_sizes) > max_predator_size:
        return None, None

    with np.errstate(over="ignore", divide="ignore"):
        # 1. Calculate the m_ij matrix (base^intersection_size)
        power_matrix = base**intersection_sizes

        # 2. Get n_i as a column vector for broadcasting
        n_i = np.array(sv_sizes, dtype=float)[:, np.newaxis]

        # 3. Calculate denominator (n_i * base^m_ij)
        # Avoid division by zero if n_i is 0 (though unlikely for non-contradictory SVs)
        n_i[n_i == 0] = 1e-9
        denominator = n_i * power_matrix

        # 4. Calculate scores matrix
        scores_matrix = 1.0 / denominator

    # --- add weight to take prey size into account ---
    # weight = 1 + 2 * (1 - np.exp(- 0.01 * np.array(sv_sizes)))
    # scores_matrix = scores_matrix ** weight  # increase scores that are > 1, and decrease those that are < 1

    # 5. Filter for scores > 1
    np.fill_diagonal(scores_matrix, 0)  # A vector cannot be its own prey
    scores_gt_1 = scores_matrix * (scores_matrix > threshold)

    # 6. Calculate row-scores
    row_scores = np.mean(scores_gt_1**2, axis=1)
    # row_scores = np.sum(scores_gt_1 ** 2, axis=1)
    # row_scores = np.sum(scores_gt_1, axis=1)

    # --- Apply predator size constraint ---
    # Create a mask for predators that are small enough
    sv_sizes_array = np.array(sv_sizes)
    predator_size_mask = sv_sizes_array <= max_predator_size

    # Apply the mask: set scores of large predators to 0
    masked_row_scores = row_scores * predator_size_mask
    # --- End constraint ---

    # 7. Find the best predator
    best_row_index = np.argmax(masked_row_scores)
    best_score = masked_row_scores[best_row_index]

    if best_score > 0:
        # Find prey for this predator
        prey_indices = np.where(scores_gt_1[best_row_index] > 0)[0]

        if prey_indices.size > 0:
            prey_indices_list = prey_indices.astype(int).tolist()
            return int(best_row_index), prey_indices_list

    # No predator found
    return None, None
