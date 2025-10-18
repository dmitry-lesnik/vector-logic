"""
Tests for the RuleParser class.
"""

import pytest

from rule_engine.rule_parser import RuleParser


@pytest.fixture
def variable_map():
    """Provides a sample variable map for testing."""
    return {"x1": 1, "x2": 2, "x3": 3, "x4": 4}


def test_parse_single_variable(variable_map):
    """Tests parsing a single, non-negated variable."""
    parser = RuleParser(variable_map)
    ast = parser.parse("x1")
    assert ast == ("var", False, "x1")


def test_parse_negated_variable(variable_map):
    """Tests parsing a single, negated variable."""
    parser = RuleParser(variable_map)
    ast = parser.parse("!x2")
    assert ast == ("var", True, "x2")


def test_parse_simple_binary_operations(variable_map):
    """Tests parsing of all simple binary operations."""
    parser = RuleParser(variable_map)
    operations = ["&&", "||", "^^", "=>", "<=", "="]
    for op in operations:
        rule = f"x1 {op} x2"
        ast = parser.parse(rule)
        expected_ast = ("op", op, ("var", False, "x1"), ("var", False, "x2"))
        assert ast == expected_ast


def test_parse_equivalence_operator(variable_map):
    """Tests parsing of the equivalence operator '<=>'."""
    parser = RuleParser(variable_map)
    rule = "x1 <=> x2"
    ast = parser.parse(rule)
    expected_ast = ("op", "=", ("var", False, "x1"), ("var", False, "x2"))
    assert ast == expected_ast


def test_precedence_and_over_or(variable_map):
    """Tests that AND (&&) has higher precedence than OR (||)."""
    parser = RuleParser(variable_map)
    # Should parse as x1 || (x2 && x3)
    ast = parser.parse("x1 || x2 && x3")
    expected_ast = ("op", "||", ("var", False, "x1"), ("op", "&&", ("var", False, "x2"), ("var", False, "x3")))
    assert ast == expected_ast


def test_precedence_or_over_implies(variable_map):
    """Tests that OR (||) has higher precedence than IMPLIES (=>)."""
    parser = RuleParser(variable_map)
    # Should parse as (x1 || x2) => x3 because || is higher precedence
    ast = parser.parse("x1 || x2 => x3")
    expected_ast = ("op", "=>", ("op", "||", ("var", False, "x1"), ("var", False, "x2")), ("var", False, "x3"))
    assert ast == expected_ast


def test_parentheses_override_precedence(variable_map):
    """Tests that parentheses correctly override the default operator precedence."""
    parser = RuleParser(variable_map)
    # Should parse as (x1 || x2) && x3
    ast = parser.parse("(x1 || x2) && x3")
    expected_ast = ("op", "&&", ("op", "||", ("var", False, "x1"), ("var", False, "x2")), ("var", False, "x3"))
    assert ast == expected_ast


def test_associativity_of_operators(variable_map):
    """Tests the left-to-right associativity of operators at the same precedence level."""
    parser = RuleParser(variable_map)

    # Should parse as (x1 || x2) ^^ x3
    ast_xor = parser.parse("x1 || x2 ^^ x3")
    expected_xor = (
        "op",
        "^^",
        ("op", "||", ("var", False, "x1"), ("var", False, "x2")),
        ("var", False, "x3"),
    )
    assert ast_xor == expected_xor

    # Should parse as (x1 => x2) = x3
    ast_eq = parser.parse("x1 => x2 = x3")
    expected_eq = (
        "op",
        "=",
        ("op", "=>", ("var", False, "x1"), ("var", False, "x2")),
        ("var", False, "x3"),
    )
    assert ast_eq == expected_eq


def test_negated_parentheses_raises_error(variable_map):
    """Tests that parsing a negated parenthesized expression raises an error."""
    parser = RuleParser(variable_map)
    with pytest.raises(ValueError, match="Invalid rule syntax: Negation of expressions in parentheses is not allowed."):
        parser.parse("!(x1 && x2)")


def test_complex_rule(variable_map):
    """Tests a complex rule with multiple operators, negations, and parentheses."""
    parser = RuleParser(variable_map)
    rule = "x1 || (!x2 => (x3 ^^ !x4))"
    ast = parser.parse(rule)
    expected_ast = (
        "op",
        "||",
        ("var", False, "x1"),
        ("op", "=>", ("var", True, "x2"), ("op", "^^", ("var", False, "x3"), ("var", True, "x4"))),
    )
    assert ast == expected_ast


def test_undefined_variable_raises_error(variable_map):
    """Tests that an undefined variable raises a ValueError."""
    parser = RuleParser(variable_map)
    with pytest.raises(ValueError, match="Variable 'y1' is not defined in the engine."):
        parser.parse("x1 && y1")


def test_mismatched_parentheses_raise_error(variable_map):
    """Tests that mismatched parentheses raise a ValueError."""
    parser = RuleParser(variable_map)
    with pytest.raises(ValueError, match="Invalid rule syntax"):
        parser.parse("(x1 && x2")

    with pytest.raises(ValueError, match="Invalid rule syntax"):
        parser.parse("x1 && x2)")


def test_invalid_character_raises_error(variable_map):
    """Tests that an invalid character in the rule string raises a ValueError."""
    parser = RuleParser(variable_map)
    with pytest.raises(ValueError, match="Invalid rule syntax"):
        parser.parse("x1 # x2")


def test_empty_rule_raises_error(variable_map):
    """Tests that an empty rule string raises a ValueError."""
    parser = RuleParser(variable_map)
    with pytest.raises(ValueError, match="Cannot parse an empty rule string."):
        parser.parse("")


def test_unexpected_tokens_at_end_raise_error(variable_map):
    """Tests that extra tokens at the end of a valid rule raise a ValueError."""
    parser = RuleParser(variable_map)
    with pytest.raises(ValueError, match="Invalid rule syntax"):
        parser.parse("x1 && x2 x3")
