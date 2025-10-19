"""
End-to-end tests for the RuleConverter class, combining all relevant test cases.
"""

import pytest

from vectorlogic.rule_converter import RuleConverter
from vectorlogic.state_vector import StateVector
from vectorlogic.t_object import TObject


@pytest.fixture
def variable_map():
    """Provides a sample variable map for testing."""
    return {"x1": 1, "x2": 2, "x3": 3, "x4": 4, "x5": 5}


@pytest.fixture
def converter(variable_map):
    """Provides an instance of the RuleConverter."""
    return RuleConverter(variable_map)


@pytest.fixture
def binary_map():
    op2vector = {
        "&&": StateVector([TObject(ones={1, 2})]),
        "||": StateVector([TObject(ones={1}), TObject(ones={2}, zeros={1})]),
        "^^": StateVector([TObject(ones={1}, zeros={2}), TObject(ones={2}, zeros={1})]),
        "=>": StateVector([TObject(ones={1, 2}), TObject(zeros={1})]),
        "<=": StateVector([TObject(ones={1}), TObject(zeros={1, 2})]),
        "=": StateVector([TObject(ones={1, 2}), TObject(zeros={1, 2})]),
    }
    return op2vector


@pytest.fixture
def triplet_map():
    op2vector = {
        "&&": StateVector(
            [
                TObject(ones={1, 2, 3}),
                TObject(zeros={1, 2}),
                TObject(ones={2}, zeros={1, 3}),
            ]
        ),
        "||": StateVector(
            [
                TObject(ones={1, 2}),
                TObject(ones={1, 3}, zeros={2}),
                TObject(zeros={1, 2, 3}),
            ]
        ),
        "^^": StateVector(
            [
                TObject(ones={1, 2}, zeros={3}),
                TObject(ones={1, 3}, zeros={2}),
                TObject(zeros={1, 2, 3}),
                TObject(ones={2, 3}, zeros={1}),
            ]
        ),
        "=>": StateVector(
            [
                TObject(ones={1, 2, 3}),
                TObject(ones={1}, zeros={2}),
                TObject(ones={2}, zeros={1, 3}),
            ]
        ),
        "<=": StateVector(
            [
                TObject(ones={1, 2}),
                TObject(ones={1}, zeros={2, 3}),
                TObject(ones={3}, zeros={1, 2}),
            ]
        ),
        "=": StateVector(
            [
                TObject(ones={1, 2, 3}),
                TObject(ones={1}, zeros={2, 3}),
                TObject(ones={2}, zeros={1, 3}),
                TObject(ones={3}, zeros={1, 2}),
            ]
        ),
    }
    return op2vector


def test_convert_single_variable_positive(converter):
    """Tests conversion of a single positive variable."""
    sv = converter.convert("x1")
    expected_sv = StateVector([TObject(ones={1})])
    assert sv == expected_sv


def test_convert_single_variable_negated(converter):
    """Tests conversion of a single negated variable."""
    sv = converter.convert("!x2")
    expected_sv = StateVector([TObject(zeros={2})])
    assert sv == expected_sv


def test_convert_simple_binary_operations(converter, binary_map):
    """Tests conversion of all simple binary operation rules."""
    for op, expected_sv in binary_map.items():
        sv = converter.convert(f"x1 {op} x2")
        assert sv == expected_sv


def test_convert_simple_or_operation_with_negation(converter):
    """Tests conversion of a simple 'or' operation with a negated variable."""
    sv = converter.convert("!x1 || x2")
    # Expected for !x1 || x2
    expected_sv = StateVector([TObject(zeros={1}), TObject(ones={1, 2})])
    assert sv == expected_sv


def test_convert_triplet_operations(converter, triplet_map):
    """Tests conversion of all triplet operation rules."""
    for op, expected_sv in triplet_map.items():
        sv = converter.convert(f"x1 = (x2 {op} x3)")
        assert sv == expected_sv


def test_convert_triplet_with_negation(converter):
    """Tests conversion of a triplet rule with negation."""
    sv = converter.convert("!x1 = (x2 || !x3)")
    # This is equivalent to x1 = !(x2 || !x3) which is x1 = (!x2 && x3)
    expected_sv = StateVector(
        [
            TObject(ones={2}, zeros={1}),
            TObject(zeros={1, 2, 3}),
            TObject(ones={1, 3}, zeros={2}),
        ]
    )
    assert sv == expected_sv


def test_convert_swapped_triplet_operation(converter):
    """Tests that a triplet rule is the same regardless of order."""
    sv1 = converter.convert("(x2 => x3) = x1")
    sv2 = converter.convert("x1 = (x2 => x3)")
    assert sv1 == sv2


def test_convert_complex_rule(converter):
    """Tests converting a complex rule that requires flattening."""
    sv = converter.convert("(x1 && x2) => x3")
    expected_sv = StateVector([TObject(ones={1, 2, 3}), TObject(zeros={1}), TObject(ones={1}, zeros={2})])
    assert sv == expected_sv


def test_converter_resets_state(converter):
    """
    Tests that the converter resets its auxiliary variable counter between calls.
    """
    converter.convert("(x1 && x2) => x3")
    # After the call, the internal map should reflect the variables created.
    assert converter._aux_var_map == {"__aux_1": -1}

    # A second call should start from a clean slate.
    # it creates no aux variables since this is a simple triplet
    converter.convert("(x4 || x5) = x1")
    assert converter._aux_var_map == {}

    # A more complex rule should reset and create two new aux vars.
    converter.convert("(x1 && x2) => (x3 <= x4)")
    assert converter._aux_var_map == {"__aux_1": -1, "__aux_2": -2}


def test_undefined_variable_raises_error(converter):
    """Tests that a rule with an undefined variable raises a ValueError."""
    with pytest.raises(ValueError, match="Variable 'y1' is not defined"):
        converter.convert("x1 && y1")


def test_invalid_syntax_raises_error(converter):
    """Tests that a rule with invalid syntax raises a ValueError."""
    with pytest.raises(ValueError, match="Invalid rule syntax"):
        converter.convert("x1 && (x2")


def test_convert_complex_rule_flattening_and_conversion(converter):
    """
    Tests that a complex rule is correctly converted into StateVector.
    """
    sv = converter.convert("(x1 <= !x2) => !x3")
    expected_sv1 = StateVector(
        [
            TObject(ones={1}, zeros={3}),
            TObject(ones={2}, zeros={1, 3}),
            TObject(zeros={1, 2}),
        ]
    )
    assert expected_sv1 == sv

    sv = converter.convert("x1 => (x3 || (!x4 => x5))")
    expected_sv2 = StateVector(
        [
            TObject(zeros={1, 3, 4, 5}),
            TObject(ones={3}, zeros={4, 5}),
            TObject(ones={5}, zeros={4}),
            TObject(ones={4}),
        ]
    )
    assert expected_sv2 == sv


@pytest.mark.parametrize(
    "rule, expected_sv",
    [
        ("x1 && x1", StateVector([TObject(ones={1})])),
        ("x1 || x1", StateVector([TObject(ones={1})])),
        ("x1 = (!x1)", StateVector()),
        ("x1 = (!x1 && x2)", StateVector([TObject(zeros={1, 2})])),
        ("x1 => (x2 && x1)", StateVector([TObject(ones={1, 2}), TObject(zeros={1})])),
    ],
)
def test_convert_rules_with_repeated_variables(converter, rule, expected_sv):
    """
    Tests rules with repeated variables simplify to their correct logical form.
    - (x1 && x1) should be equivalent to x1.
    - (x1 || x1) should be equivalent to x1.
    - (x1 => (x2 && x1)) should be equivalent to (!x1 || x2).
    """
    sv = converter.convert(rule)
    assert sv == expected_sv


def test_convert_repeated_variable_in_complex_rule(converter):
    """Tests repeated variables in a rule that also needs flattening."""
    sv = converter.convert("(x1 && x2) <= (x3 || (!x1))")
    expected_sv = StateVector([TObject(ones={1}, zeros={2, 3}), TObject(ones={1, 2})])
    assert sv == expected_sv

    converter.convert("(x1 || !x2) <= (x2 || (!x1 || !x2))")
    simple_asts = [
        ("op", "=", ("var", False, "__aux_4"), ("op", "||", ("var", False, "x1"), ("var", True, "x2"))),
        ("op", "=", ("var", False, "__aux_5"), ("op", "||", ("var", True, "__aux_2"), ("var", True, "__aux_3"))),
        ("op", "=", ("var", False, "__aux_6"), ("op", "||", ("var", False, "__aux_1"), ("var", False, "__aux_5"))),
        ("op", "<=", ("var", False, "__aux_4"), ("var", False, "__aux_6")),
        ("op", "=", ("var", False, "x2"), ("var", False, "__aux_1")),
        ("op", "=", ("var", False, "x1"), ("var", False, "__aux_2")),
        ("op", "=", ("var", False, "x2"), ("var", False, "__aux_3")),
    ]
    assert simple_asts == converter._all_simple_asts
