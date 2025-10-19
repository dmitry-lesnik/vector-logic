"""
Tests for the TObject class.
"""

import pytest

from vectorlogic.t_object import TObject


def test_t_object_default_initialization():
    """Tests that TObject() creates a trivial object."""
    t_obj = TObject()
    assert not t_obj.is_null
    assert t_obj.ones == frozenset()
    assert t_obj.zeros == frozenset()
    assert t_obj.is_trivial


def test_t_object_initialization_with_disjoint_sets():
    """Tests initialization with valid, disjoint 'ones' and 'zeros' sets."""
    ones_set = {1, 3, 5}
    zeros_set = {2, 4, 6}
    t_obj = TObject(ones=ones_set, zeros=zeros_set)
    assert t_obj.ones == frozenset(ones_set)
    assert t_obj.zeros == frozenset(zeros_set)
    assert not t_obj.is_null
    assert not t_obj.is_trivial


def test_t_object_initialization_from_lists():
    """Tests that list inputs are correctly converted to frozensets."""
    ones_list = [1, 5, 3, 5]
    zeros_list = [2, 4, 2, 6]
    t_obj = TObject(ones=ones_list, zeros=zeros_list)
    assert t_obj.ones == frozenset([1, 3, 5])
    assert t_obj.zeros == frozenset([2, 4, 6])


def test_non_disjoint_sets_create_null_object():
    """
    Tests that non-disjoint 'ones' and 'zeros' sets result in a null TObject.
    """
    t_obj = TObject(ones={1, 2, 3}, zeros={3, 4, 5})
    assert t_obj.is_null


def test_null_initialization():
    """Tests creating a null TObject by passing is_null=True."""
    t_obj = TObject(is_null=True)
    assert t_obj.is_null
    assert t_obj.ones == frozenset()
    assert t_obj.zeros == frozenset()


def test_invalid_initialization_raises_error():
    """
    Tests that a ValueError is raised when providing 'ones' or 'zeros'
    along with is_null=True.
    """
    match_str = "Cannot specify 'ones' or 'zeros' when 'is_null' is True."
    with pytest.raises(ValueError, match=match_str):
        TObject(ones={1, 2, 3}, is_null=True)

    with pytest.raises(ValueError, match=match_str):
        TObject(zeros={3, 4, 5}, is_null=True)

    with pytest.raises(ValueError, match=match_str):
        TObject(ones={1}, zeros={2}, is_null=True)


def test_properties():
    """Tests the property getters."""
    ones = {1, 10}
    zeros = {5, 15}
    t_obj = TObject(ones=ones, zeros=zeros)
    assert t_obj.ones == frozenset(ones)
    assert t_obj.zeros == frozenset(zeros)
    assert not t_obj.is_null


def test_is_trivial_property():
    """Tests the is_trivial property."""
    assert TObject().is_trivial
    assert not TObject(ones={1}).is_trivial
    assert not TObject(zeros={1}).is_trivial
    assert not TObject(is_null=True).is_trivial
    assert not TObject(ones={1}, zeros={1}).is_trivial


def test_object_representation():
    """Tests the __repr__ method for accurate representation."""
    assert repr(TObject(is_null=True)) == "TObject(is_null=True)"
    assert repr(TObject()) == "TObject(ones=set(), zeros=set())"
    assert repr(TObject(ones={1, 2}, zeros={2, 3})) == "TObject(is_null=True)"
    t_obj_regular = TObject(ones={1}, zeros={2})
    assert "TObject(" in repr(t_obj_regular)
    assert "ones={1}" in repr(t_obj_regular)
    assert "zeros={2}" in repr(t_obj_regular)


def test_to_string_method():
    """Tests the to_string method for various TObject states."""
    assert TObject(is_null=True).to_string() == "null"
    assert TObject().to_string() == "---"
    assert TObject(ones={1, 4}, zeros={3}).to_string() == "1 - 0 1"
    assert TObject(ones={2}, zeros={5}).to_string(max_index=6) == "- 1 - - 0 -"
    assert TObject(ones={2, 6}, zeros={3}).to_string(max_index=4) == "- 1 0 -"
    assert TObject(ones={3}).to_string() == "- - 1"
    assert TObject(zeros={2, 4}).to_string() == "- 0 - 0"


def test_multiplication():
    """Tests the multiplication of TObjects."""
    t_a = TObject(ones={1, 2}, zeros={3, 4})
    t_b = TObject(ones={5}, zeros={6})
    t_null = TObject(is_null=True)
    t_trivial = TObject()

    assert (t_a * t_null).is_null
    assert (t_null * t_a).is_null
    assert (t_null * t_null).is_null

    res_trivial = t_a * t_trivial
    assert res_trivial == t_a

    res_trivial_rev = t_trivial * t_a
    assert res_trivial_rev == t_a

    res = t_a * t_b
    assert res.ones == frozenset([1, 2, 5])
    assert res.zeros == frozenset([3, 4, 6])
    assert not res.is_null

    t_c = TObject(ones={3}, zeros={7})
    res_conflict = t_a * t_c
    assert res_conflict.is_null

    res_idem = t_a * t_a
    assert res_idem == t_a


def test_reduction():
    """Tests the reduce method for TObjects."""
    t1 = TObject(ones={1, 2}, zeros={3})
    t2 = TObject(ones={1}, zeros={2, 3})
    reduced = t1.reduce(t2)
    assert reduced == TObject(ones={1}, zeros={3})

    t3 = TObject(ones={5}, zeros={6, 7})
    t4 = TObject(ones={5, 7}, zeros={6})
    reduced_rev = t3.reduce(t4)
    assert reduced_rev == TObject(ones={5}, zeros={6})
    assert t4.reduce(t3) == reduced_rev

    t5 = TObject(ones={10}, zeros={11})
    t6 = TObject(ones={12}, zeros={13})
    assert t5.reduce(t6) is None

    t7 = TObject(ones={20, 21}, zeros={22})
    t8 = TObject(ones={20}, zeros={21, 22})
    reduced_case_4 = t7.reduce(t8)
    assert reduced_case_4 == TObject(ones={20}, zeros={22})

    t9 = TObject(ones={30, 31}, zeros={32})
    t10 = TObject(ones={30}, zeros={31, 33})
    assert t9.reduce(t10) is None

    t11 = TObject(ones={40, 41}, zeros={42})
    t12 = TObject(ones={42}, zeros={40, 41})
    assert t11.reduce(t12) is None


def test_is_superset():
    """Tests the is_superset method for TObjects."""
    t1 = TObject(ones={1, 2}, zeros={3, 4})
    t2 = TObject(ones={1}, zeros={3})
    assert t1.is_superset(t2) == -1
    assert t2.is_superset(t1) == 1

    t3 = TObject(ones={1}, zeros={3})
    t4 = TObject(ones={1, 2}, zeros={3, 4})
    assert t3.is_superset(t4) == 1
    assert t4.is_superset(t3) == -1

    t5 = TObject(ones={1, 2}, zeros={3})
    t6 = TObject(ones={1}, zeros={3, 4})
    assert t5.is_superset(t6) == 0
    assert t6.is_superset(t5) == 0

    t7 = TObject(ones={1}, zeros={2})
    t8 = TObject(ones={1}, zeros={2})
    assert t7.is_superset(t8) == 1

    t9 = TObject(ones={1}, zeros={2})
    t10 = TObject()
    assert t9.is_superset(t10) == -1
    assert t10.is_superset(t9) == 1

    t11 = TObject()
    t12 = TObject()
    assert t11.is_superset(t12) == 1


def test_negate_variables():
    """Tests the negation of variables in a TObject."""
    t1 = TObject(ones={1, 2}, zeros={3, 4})
    negated_t1 = t1.negate_variables([1, 3, 5])
    assert negated_t1.ones == frozenset([2, 3])
    assert negated_t1.zeros == frozenset([1, 4])
    assert t1.ones == frozenset([1, 2])
    assert t1.zeros == frozenset([3, 4])

    t2 = TObject(ones={1})
    negated_t2 = t2.negate_variables(1)
    assert negated_t2.ones == frozenset()
    assert negated_t2.zeros == frozenset([1])

    t3 = TObject(ones={1, 2, 3}, zeros={4, 5, 6})
    negated_t3 = t3.negate_variables([1, 4, 7, 2, 5])
    assert negated_t3.ones == frozenset([3, 4, 5])
    assert negated_t3.zeros == frozenset([1, 2, 6])

    t4 = TObject()
    negated_t4 = t4.negate_variables([1, 2])
    assert negated_t4.ones == frozenset()
    assert negated_t4.zeros == frozenset()

    t5 = TObject(is_null=True)
    negated_t5 = t5.negate_variables([1])
    assert negated_t5.is_null

    t6 = TObject(ones={1}, zeros={2})
    negated_t6 = t6.negate_variables([])
    assert negated_t6 == t6


def test_remove_variables():
    """Tests the removal of variables from a TObject."""
    t1 = TObject(ones={1, 2}, zeros={3, 4})
    removed_t1 = t1.remove_variables([1, 3, 5])
    assert removed_t1.ones == frozenset([2])
    assert removed_t1.zeros == frozenset([4])
    assert t1.ones == frozenset([1, 2])
    assert t1.zeros == frozenset([3, 4])

    t2 = TObject(ones={1})
    removed_t2 = t2.remove_variables(1)
    assert removed_t2.is_trivial

    t3 = TObject(is_null=True)
    removed_t3 = t3.remove_variables([1])
    assert removed_t3.is_null
