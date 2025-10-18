"""
Tests for the StateVector class.
"""

import pytest

from rule_engine.state_vector import StateVector
from rule_engine.t_object import TObject


def test_state_vector_default_initialization():
    """Tests that a StateVector can be initialized with no arguments."""
    sv = StateVector()
    assert sv._t_objects == ()


def test_state_vector_initialization_with_t_objects():
    """Tests that a StateVector can be initialized with a list of TObjects."""
    t_obj1 = TObject(ones={1})
    t_obj2 = TObject(zeros={2})
    sv = StateVector(t_objects=[t_obj1, t_obj2])

    assert len(sv._t_objects) == 2
    assert sv._t_objects[0] is t_obj1
    assert sv._t_objects[1] is t_obj2


def test_equality():
    """Tests the equality comparison of StateVectors."""
    t1 = TObject(ones={1})
    t2 = TObject(zeros={2})
    sv1 = StateVector([t1, t2])
    sv2 = StateVector([t2, t1])  # Same objects, different order
    sv3 = StateVector([t1, TObject(zeros={3})])

    assert sv1 == sv2
    assert sv1 != sv3


def test_to_string_method():
    """Tests the to_string method of the StateVector."""

    # Case 1: Empty state vector
    sv_empty = StateVector()
    assert sv_empty.to_string() == "{ Contradiction }"

    # Case 2: Auto-calculated max_index
    t1 = TObject(ones={1}, zeros={3})
    t2 = TObject(ones={4})
    sv = StateVector([t1, t2])
    # max_index should be 4
    expected_auto = "{\n    1 - 0 -\n    - - - 1\n}"
    assert sv.to_string() == expected_auto

    # Case 3: Explicit max_index (larger)
    expected_explicit = "{\n    1 - 0 - -\n    - - - 1 -\n}"
    assert sv.to_string(max_index=5) == expected_explicit

    # Case 4: With indent
    expected_indent = "  {\n      1 - 0 -\n      - - - 1\n  }"
    assert sv.to_string(indent=2) == expected_indent

    # Case 5: State vector with a null object
    t_null = TObject(is_null=True)
    sv_with_null = StateVector([t1, t_null])
    # max_index should be 3 from t1
    expected_with_null = "{\n    1 - 0\n    null\n}"
    assert sv_with_null.to_string() == expected_with_null

    # Case 6: State vector with only a trivial object
    sv_trivial = StateVector([TObject()])
    # max_index should be 0, to_string of TObject handles this
    expected_trivial = "{\n    ---\n}"
    assert sv_trivial.to_string() == expected_trivial

    # Case 7: Empty state vector with indent
    assert sv_empty.to_string(indent=4) == "    { Contradiction }"


def test_state_vector_multiplication():
    """Tests the multiplication of StateVectors and the subsequent reduction."""

    # Scenario 1: Standard multiplication (no reduction needed)
    t1 = TObject(ones={1}, zeros={2})
    t2 = TObject(ones={3}, zeros={4})
    sv1 = StateVector([t1])
    sv2 = StateVector([t2])
    result1 = sv1 * sv2
    assert len(result1._t_objects) == 1
    expected_t1 = TObject(ones={1, 3}, zeros={2, 4})
    assert result1._t_objects[0] == expected_t1

    # Scenario 2: Multiplication with null products (no reduction needed)
    t3 = TObject(ones={1}, zeros={2})
    t4 = TObject(ones={2}, zeros={3})  # This will produce a null product with t3
    t5 = TObject(ones={4}, zeros={5})
    sv3 = StateVector([t3, t5])
    sv4 = StateVector([t4])
    result2 = sv3 * sv4
    assert len(result2._t_objects) == 1
    expected_t2 = TObject(ones={2, 4}, zeros={3, 5})
    assert result2._t_objects[0] == expected_t2

    # Scenario 3: Multiplication by an empty state vector
    sv5 = StateVector([TObject(ones={1})])
    sv_empty = StateVector()
    result3 = sv5 * sv_empty
    assert len(result3._t_objects) == 0

    # Scenario 4: Multiplication of two non-empty state vectors (no reduction needed)
    t6 = TObject(ones={1})
    t7 = TObject(zeros={2})
    t8 = TObject(ones={3})
    t9 = TObject(zeros={4})
    sv6 = StateVector([t6, t7])
    sv7 = StateVector([t8, t9])
    result4 = sv6 * sv7
    assert len(result4._t_objects) == 4
    expected_products = [
        TObject(ones={1, 3}),
        TObject(ones={1}, zeros={4}),
        TObject(ones={3}, zeros={2}),
        TObject(zeros={2, 4}),
    ]
    for p in expected_products:
        assert p in result4._t_objects

    # Scenario 5: Multiplication with reduction
    sv_a = StateVector([TObject(ones={1}), TObject(zeros={1})])
    sv_b = StateVector([TObject(ones={2})])
    result5 = sv_a * sv_b
    assert len(result5._t_objects) == 1
    expected_t5 = TObject(ones={2})
    assert result5._t_objects[0] == expected_t5

    sv1 = StateVector([TObject(ones={1}), TObject(zeros={2})])
    sv2 = StateVector([TObject(ones={3}), TObject(zeros={4})])

    product = sv1 * sv2

    expected_sv = StateVector(
        [
            TObject(ones={1, 3}),
            TObject(ones={1}, zeros={4}),
            TObject(ones={3}, zeros={2}),
            TObject(zeros={2, 4}),
        ]
    )
    assert product == expected_sv


def test_state_vector_reduce_basic():
    """Tests a basic reduction in StateVector."""
    t1 = TObject(ones={1}, zeros={2, 3})
    t2 = TObject(ones={1, 3}, zeros={2})
    sv = StateVector([t1, t2])
    simplified_sv = sv.simplify()
    assert len(simplified_sv._t_objects) == 1
    assert simplified_sv._t_objects[0] == TObject(ones={1}, zeros={2})
    assert len(sv._t_objects) == 2  # Original is unchanged


def test_state_vector_no_reduction():
    """Tests StateVector when no reduction is possible."""
    t1 = TObject(ones={1})
    t2 = TObject(zeros={2})
    sv = StateVector([t1, t2])
    simplified_sv = sv.simplify()
    assert len(simplified_sv._t_objects) == 2
    assert sv == simplified_sv  # Should be equal as no change


def test_state_vector_multiple_reductions():
    """Tests multiple reductions in a single pass."""
    t1 = TObject(ones={1}, zeros={2, 3})
    t2 = TObject(ones={1, 3}, zeros={2})
    t3 = TObject(ones={4}, zeros={5, 6})
    t4 = TObject(ones={4, 6}, zeros={5})
    sv = StateVector([t1, t2, t3, t4])
    simplified_sv = sv.simplify()
    assert len(simplified_sv._t_objects) == 2
    # Note: The order of reduced objects is not guaranteed
    assert TObject(ones={1}, zeros={2}) in simplified_sv._t_objects
    assert TObject(ones={4}, zeros={5}) in simplified_sv._t_objects
    assert len(sv._t_objects) == 4  # Original is unchanged


def test_state_vector_sequential_reductions():
    """Tests reductions that occur over multiple iterations."""
    t1 = TObject(ones={1, 4}, zeros={2, 3})
    t2 = TObject(ones={1, 3, 4}, zeros={2})
    t3 = TObject(ones={1}, zeros={2, 4})
    sv = StateVector([t1, t2, t3])
    simplified_sv = sv.simplify()

    assert len(simplified_sv._t_objects) == 1
    assert simplified_sv._t_objects[0] == TObject(ones={1}, zeros={2})
    assert len(sv._t_objects) == 3  # Original is unchanged


def test_state_vector_full_reduction():
    """Tests both subsumption and adjacency reduction in StateVector."""
    # t2 is a superset of t1, so t2 should be removed by subsumption
    t1 = TObject(ones={1}, zeros={2})
    t2 = TObject(ones={1, 3}, zeros={2})

    # t3 and t4 are reducible by adjacency
    t3 = TObject(ones={4}, zeros={5, 6})
    t4 = TObject(ones={4, 6}, zeros={5})

    # t5 is a superset of the result of t3 and t4's reduction
    t5 = TObject(ones={4}, zeros={5, 7})

    sv = StateVector([t1, t2, t3, t4, t5])
    simplified_sv = sv.simplify(reduce_subsumption=True)

    assert len(simplified_sv._t_objects) == 2

    # Check that the correct objects remain
    expected_t1 = TObject(ones={1}, zeros={2})
    expected_t_reduced = TObject(ones={4}, zeros={5})

    assert expected_t1 in simplified_sv._t_objects
    assert expected_t_reduced in simplified_sv._t_objects
    assert len(sv._t_objects) == 5  # Original is unchanged


def test_reduction_cleaning_steps():
    """Tests the cleaning steps in the simplify method."""

    # Scenario 1: Removal of null objects
    t1 = TObject(ones={1})
    t_null = TObject(is_null=True)
    sv = StateVector([t1, t_null, TObject(ones={2})])
    simplified_sv = sv.simplify()
    assert len(simplified_sv._t_objects) == 2
    assert t_null not in simplified_sv._t_objects
    assert t1 in simplified_sv._t_objects
    assert TObject(ones={2}) in simplified_sv._t_objects

    # Scenario 2: Handling of trivial objects
    t2 = TObject(ones={1})
    t_trivial = TObject()
    sv = StateVector([t2, t_trivial, TObject(zeros={3})])
    simplified_sv = sv.simplify()
    assert len(simplified_sv._t_objects) == 1
    assert simplified_sv._t_objects[0].is_trivial

    # Scenario 3: Removal of duplicate objects
    t3 = TObject(ones={1}, zeros={2})
    t4 = TObject(ones={3}, zeros={4})
    sv = StateVector([t3, t4, t3, t3, t4])
    simplified_sv = sv.simplify()
    assert len(simplified_sv._t_objects) == 2
    assert simplified_sv._t_objects.count(t3) == 1
    assert simplified_sv._t_objects.count(t4) == 1
    # Order is not guaranteed after simplify due to dict.fromkeys and set operations
    assert t3 in simplified_sv._t_objects
    assert t4 in simplified_sv._t_objects


def test_state_vector_negate_variables():
    """Tests the negate_variables method of the StateVector."""

    # Scenario 1: Basic negation on a single TObject
    t1 = TObject(ones={1, 2}, zeros={3, 4})
    sv1 = StateVector([t1])
    negated_sv1 = sv1.negate_variables([1, 3, 5])  # Negate one from 'ones', one from 'zeros', one not present

    expected_t1 = TObject(ones={2, 3}, zeros={1, 4})
    assert len(negated_sv1._t_objects) == 1
    assert negated_sv1._t_objects[0] == expected_t1
    assert sv1._t_objects[0] == t1  # Original is unchanged

    # Scenario 2: Negation on multiple TObjects
    t2 = TObject(ones={1}, zeros={2})
    t3 = TObject(ones={2}, zeros={3})
    sv2 = StateVector([t2, t3])
    negated_sv2 = sv2.negate_variables([1, 2])

    expected_t2 = TObject(ones={2}, zeros={1})
    expected_t3 = TObject(zeros={2, 3})
    assert len(negated_sv2._t_objects) == 2
    # The order of objects is preserved
    assert negated_sv2._t_objects[0] == expected_t2
    assert negated_sv2._t_objects[1] == expected_t3
    assert sv2._t_objects[0] == t2  # Original is unchanged
    assert sv2._t_objects[1] == t3  # Original is unchanged

    # Scenario 3: Negating with an empty set of variables
    t4 = TObject(ones={1}, zeros={2})
    sv3 = StateVector([t4])
    original_t4 = TObject(ones={1}, zeros={2})
    negated_sv3 = sv3.negate_variables([])

    assert len(negated_sv3._t_objects) == 1
    assert negated_sv3._t_objects[0] == original_t4
    assert sv3 == negated_sv3

    # Scenario 4: Negating on an empty StateVector
    sv_empty = StateVector()
    negated_sv_empty = sv_empty.negate_variables([1, 2, 3])

    assert len(negated_sv_empty._t_objects) == 0
    assert sv_empty == negated_sv_empty


def test_state_vector_remove_variables():
    """Tests the remove_variables method of the StateVector."""
    t1 = TObject(ones={1, 2}, zeros={3, 4})
    t2 = TObject(ones={1, 5}, zeros={2, 6})
    sv = StateVector([t1, t2])
    removed_sv = sv.remove_variables([1, 3, 5])

    expected_t1 = TObject(ones={2}, zeros={4})
    expected_t2 = TObject(zeros={2, 6})

    assert removed_sv._t_objects[0] == expected_t1
    assert removed_sv._t_objects[1] == expected_t2
    assert sv._t_objects[0] == t1  # Original is unchanged
    assert sv._t_objects[1] == t2  # Original is unchanged


def test_full_reduction():
    """Tests the full reduction method, including all sub-steps."""

    sv = StateVector(
        [
            TObject(ones={1, 2}),  # Will be reduced with the one below
            TObject(ones={1}, zeros={2}),
            TObject(ones={1, 2, 3}),  # Will be subsumed by the reduced TObject
            TObject(is_null=True),  # Will be removed
        ]
    )

    simplified_sv = sv.simplify(reduce_subsumption=True)

    expected_sv = StateVector([TObject(ones={1})])
    assert simplified_sv == expected_sv


def test_var_value_all_are_one():
    """Tests var_value when all TObjects consistently define the index as 1."""
    sv = StateVector([TObject(ones={1, 2}), TObject(ones={1, 3})])
    assert sv.var_value(1) == 1


def test_var_value_all_are_zero():
    """Tests var_value when all TObjects consistently define the index as 0."""
    sv = StateVector([TObject(zeros={1, 2}), TObject(zeros={1, 3})])
    assert sv.var_value(1) == 0


def test_var_value_mixed_one_and_zero():
    """Tests var_value when TObjects have conflicting values (1 and 0)."""
    sv = StateVector([TObject(ones={1}), TObject(zeros={1})])
    assert sv.var_value(1) == -1


def test_var_value_with_dont_care():
    """
    Tests var_value when at least one TObject has undefined (-1) state
    for the index. The result should be -1.
    """
    sv = StateVector([TObject(ones={1}), TObject(ones={2})])  # Second TObject is -1 for index 1
    assert sv.var_value(1) == -1


def test_var_value_empty_vector():
    """Tests var_value on an empty StateVector."""
    sv = StateVector()
    assert sv.is_contradiction()
    with pytest.raises(ValueError, match="Cannot determine variable value for an empty StateVector"):
        sv.var_value(1)


def test_var_value_single_tobject():
    """Tests var_value on a StateVector with a single TObject."""
    sv_one = StateVector([TObject(ones={1})])
    assert sv_one.var_value(1) == 1

    sv_zero = StateVector([TObject(zeros={1})])
    assert sv_zero.var_value(1) == 0

    sv_dont_care = StateVector([TObject(ones={2})])
    assert sv_dont_care.var_value(1) == -1
