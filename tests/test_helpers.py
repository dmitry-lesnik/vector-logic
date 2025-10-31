"""
Tests for the helper functions in vectorlogic.helpers.
"""

import numpy as np
import pytest

from vectorlogic.helpers import (
    calc_ps_unions_intersections,
    update_ps_unions_intersections,
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
    print()
    print(updated_unions)
    print(updated_intersections)
    # The function should produce 1x1 matrices for the single remaining set
    np.testing.assert_array_equal(updated_unions, np.array([[3]]))
    np.testing.assert_array_equal(updated_intersections, np.array([[3]]))
