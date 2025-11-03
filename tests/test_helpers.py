"""
Tests for the helper functions in vectorlogic.helpers.
"""

import numpy as np
import pytest

from vectorlogic.helpers import (
    calc_ps_unions_intersections,
    update_ps_unions_intersections,
    find_next_cluster,
    find_predator_prey,
)


@pytest.fixture
def sample_pivot_sets():
    """Provides a sample list of pivot sets for testing."""
    return [{1, 2, 3}, {3, 4, 5}, {1, 4, 5}]


def test_calc_ps_unions_intersections_basic(sample_pivot_sets):
    """
    Tests the calculation of union and intersection sizes for a basic case.
    """
    union_sizes, intersection_sizes = calc_ps_unions_intersections(sample_pivot_sets)

    expected_intersections = np.array([[3, 1, 1], [1, 3, 2], [1, 2, 3]])

    expected_unions = np.array([[3, 5, 5], [5, 3, 4], [5, 4, 3]])

    np.testing.assert_array_equal(intersection_sizes, expected_intersections)
    np.testing.assert_array_equal(union_sizes, expected_unions)


def test_calc_ps_unions_intersections_empty_input():
    """
    Tests that calc_ps_unions_intersections handles an empty list of pivot sets.
    """
    union_sizes, intersection_sizes = calc_ps_unions_intersections([])
    assert union_sizes.shape == (1, 0)
    assert intersection_sizes.shape == (1, 0)


def test_calc_ps_unions_intersections_disjoint_sets():
    """
    Tests the calculation with completely disjoint pivot sets.
    """
    pivot_sets = [{1, 2}, {3, 4}, {5, 6}]
    union_sizes, intersection_sizes = calc_ps_unions_intersections(pivot_sets)

    expected_intersections = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    expected_unions = np.array([[2, 4, 4], [4, 2, 4], [4, 4, 2]])

    np.testing.assert_array_equal(intersection_sizes, expected_intersections)
    np.testing.assert_array_equal(union_sizes, expected_unions)


def test_calc_ps_unions_intersections_empty_sets():
    """
    Tests the calculation with completely disjoint pivot sets.
    """
    pivot_sets = [{1, 2}, set(), {2, 3}]
    union_sizes, intersection_sizes = calc_ps_unions_intersections(pivot_sets)

    expected_unions = np.array([[2, 2, 3], [2, 0, 2], [3, 2, 2]])
    expected_intersections = np.array([[2, 0, 1], [0, 0, 0], [1, 0, 2]])

    np.testing.assert_array_equal(intersection_sizes, expected_intersections)
    np.testing.assert_array_equal(union_sizes, expected_unions)

    pivot_sets = [set(), set(), set()]
    union_sizes, intersection_sizes = calc_ps_unions_intersections(pivot_sets)

    expected_unions = np.zeros((3, 3), dtype=int)
    expected_intersections = np.zeros((3, 3), dtype=int)

    np.testing.assert_array_equal(intersection_sizes, expected_intersections)
    np.testing.assert_array_equal(union_sizes, expected_unions)


def test_update_ps_unions_intersections(sample_pivot_sets):
    """
    Tests the update logic for the union and intersection matrices
    """

    # 1. Initial calculation
    initial_unions, initial_intersections = calc_ps_unions_intersections(sample_pivot_sets)

    # 2. Simulate two added pivot sets
    new_pivot_sets = [{2, 3, 6}, {1, 6}]

    # The new list of pivot sets for the next iteration.
    updated_pivot_sets = sample_pivot_sets + new_pivot_sets

    # Indices to remove from the original matrices (sorted descending).
    indices_to_remove = [1]
    indices_to_remove = sorted(indices_to_remove, reverse=True)
    for i in indices_to_remove:
        updated_pivot_sets.pop(i)

    expected_unions, expected_intersections = calc_ps_unions_intersections(updated_pivot_sets)

    # 3. Perform the update
    updated_unions, updated_intersections = update_ps_unions_intersections(
        initial_unions, initial_intersections, indices_to_remove, updated_pivot_sets
    )

    np.testing.assert_array_equal(updated_intersections, expected_intersections)
    np.testing.assert_array_equal(updated_unions, expected_unions)


def test_update_ps_unions_intersections_no_remaining():
    """
    Tests that updating with all indices removed results in empty matrices.
    This simulates the final step of a multiplication process.
    """
    initial_unions = np.array([[2, 1], [1, 2]])
    initial_intersections = np.array([[2, 1], [1, 2]])
    new_pivot_set = {1, 2, 3}

    updated_unions, updated_intersections = update_ps_unions_intersections(
        initial_unions, initial_intersections, [1, 0], [new_pivot_set]
    )
    # print()
    # print(updated_unions)
    # print(updated_intersections)
    # The function should produce 1x1 matrices for the single remaining set
    np.testing.assert_array_equal(updated_unions, np.array([[3]]))
    np.testing.assert_array_equal(updated_intersections, np.array([[3]]))


def test_find_next_cluster(sample_pivot_sets):
    """
    Tests the find_next_cluster logic.
    """
    union_sizes, intersection_sizes = calc_ps_unions_intersections(sample_pivot_sets)

    # With max_cluster_size = 2
    # Jaccard similarities:
    # (0,1): 1/5 = 0.2
    # (0,2): 1/5 = 0.2
    # (1,2): 2/4 = 0.5
    # Row scores (max^2):
    # 0: 0.2^2 = 0.04
    # 1: 0.5^2 = 0.25
    # 2: 0.5^2 = 0.25
    # Best row is 1 (or 2). Let's say 1.
    # Scores in row 1: [0.2, 0, 0.5]
    # Sorted indices (desc): [2, 0]
    # Cluster: [1, 2]
    cluster = find_next_cluster(sample_pivot_sets, union_sizes, intersection_sizes, max_cluster_size=2)
    assert sorted(cluster) == [1, 2]

    # With max_cluster_size = 3
    # Same as above, but we take more from sorted_indices
    # Cluster: [1, 2, 0]
    cluster = find_next_cluster(sample_pivot_sets, union_sizes, intersection_sizes, max_cluster_size=3)
    assert sorted(cluster) == [0, 1, 2]


def test_find_predator_prey_success():
    """
    Tests a scenario where a predator should be successfully identified.
    """
    # sizes:     [ 1,   100,  100,  100]
    sv_sizes = [1, 100, 100, 100]
    # int_sizes: [[ 1, 1, 1, 1],
    #             [ 1, 10, 2, 2],
    #             [ 1, 2, 10, 2],
    #             [ 1, 2, 2, 10]]
    intersection_sizes = np.array([[1, 1, 1, 1], [1, 10, 2, 2], [1, 2, 10, 2], [1, 2, 2, 10]])

    # Let's analyze row 0 (the predator, size=1) with base=0.7
    # score(0,1) = 1 / (1 * 0.7^1) = 1.42
    # score(0,2) = 1 / (1 * 0.7^1) = 1.42
    # score(0,3) = 1 / (1 * 0.7^1) = 1.42
    # All scores > 1. Let's use threshold = 1.2
    # Row 0 scores > 1.2: [1.42, 1.42, 1.42]
    # Row score (mean of squares): > 0
    # Predator: 0. Prey: [1, 2, 3]

    predator_idx, prey_indices = find_predator_prey(
        sv_sizes, intersection_sizes, base=0.7, threshold=1.2, max_predator_size=2
    )

    assert predator_idx == 0
    assert sorted(prey_indices) == [1, 2, 3]


def test_find_predator_prey_no_good_predator():
    """
    Tests a scenario where no predator's scores are high enough.
    """
    # sizes:     [ 1,   100,  100]
    sv_sizes = [1, 100, 100]
    # int_sizes: [[ 1, 0, 0],
    #             [ 0, 10, 5],
    #             [ 0, 5, 10]]
    intersection_sizes = np.array([[1, 0, 0], [0, 10, 5], [0, 5, 10]])

    # Analyze row 0 (the predator, size=1) with base=0.7
    # score(0,1) = 1 / (1 * 0.7^0) = 1.0
    # score(0,2) = 1 / (1 * 0.7^0) = 1.0
    # Scores are not > 1.2 threshold.
    # Other rows are too large to be predators.
    # Result: (None, None)

    predator_idx, prey_indices = find_predator_prey(
        sv_sizes, intersection_sizes, base=0.7, threshold=1.2, max_predator_size=2
    )

    assert predator_idx is None
    assert prey_indices is None


def test_find_predator_prey_predator_too_large():
    """
    Tests that a potential predator is disqualified due to max_predator_size.
    """
    # sizes:     [ 5,   100,  100,  100]  (Predator size 5 > 2)
    sv_sizes = [5, 100, 100, 100]
    # int_sizes: [[ 5, 4, 4, 4],
    #             [ 4, 10, 2, 2],
    #             [ 4, 2, 10, 2],
    #             [ 4, 2, 2, 10]]
    intersection_sizes = np.array([[5, 4, 4, 4], [4, 10, 2, 2], [4, 2, 10, 2], [4, 2, 2, 10]])

    # Analyze row 0 (the predator, size=5) with base=0.7
    # score(0,1) = 1 / (5 * 0.7^4) = 1 / (5 * 0.24) = 0.83
    # ...scores are low anyway, but even if they were high,
    # the predator_size_mask would set row 0's score to 0.

    predator_idx, prey_indices = find_predator_prey(
        sv_sizes, intersection_sizes, base=0.7, threshold=1.2, max_predator_size=2
    )

    assert predator_idx is None
    assert prey_indices is None


def test_find_predator_prey_too_few_vectors():
    """
    Tests that the function returns (None, None) if there are too few vectors.
    """
    sv_sizes = [1, 100]
    intersection_sizes = np.array([[1, 1], [1, 10]])

    predator_idx, prey_indices = find_predator_prey(sv_sizes, intersection_sizes, max_predator_size=2)

    assert predator_idx is None
    assert prey_indices is None
