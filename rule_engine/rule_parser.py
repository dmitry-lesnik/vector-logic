"""
This module defines a high-performance RuleParser class using the pyparsing library.
"""

from typing import Dict, Any

import pyparsing as pp
from pyparsing import infixNotation, opAssoc

# --- PERFORMANCE OPTIMIZATION ---
# Enable Packrat Caching globally for pyparsing.
pp.ParserElement.enablePackrat()


# --- AST Transformer (Abstract Syntax Tree) ---
# This class converts the nested list structure from pyparsing
# into the consistent tuple-based AST used by the rest of the engine.


class AstTransformer:
    """Transforms the pyparsing result into a standardized Abstract Syntax Tree (AST)."""

    def __init__(self, variable_map: Dict[str, int]):
        self._variable_map = variable_map

    def transform_variable(self, tokens: pp.ParseResults) -> Any:
        """Transforms a variable token."""
        var_name = tokens[0]
        if var_name not in self._variable_map:
            raise ValueError(f"Variable '{var_name}' is not defined in the engine.")
        return "var", False, var_name

    @staticmethod
    def transform_unary_op(tokens: pp.ParseResults) -> Any:
        """Transforms a unary operation (negation)."""
        op, operand = tokens[0]
        if isinstance(operand, tuple) and operand[0] == "var":
            return "var", True, operand[2]
        # It's a negated expression, which is not allowed.
        raise ValueError("Negation of expressions in parentheses is not allowed.")

    @staticmethod
    def transform_binary_op(tokens: pp.ParseResults) -> Any:
        """Transforms a binary operation."""
        tokens = tokens[0]
        node = tokens[0]
        for i in range(1, len(tokens), 2):
            op, right = tokens[i], tokens[i + 1]
            if op == "<=>":
                op = "="
            node = ("op", op, node, right)
        return node


class RuleParser:
    """
    Parses rule strings into an Abstract Syntax Tree (AST) using pyparsing.
    """

    def __init__(self, variable_map: Dict[str, int]):
        """
        Initializes the RuleParser and builds the grammar.

        Args:
            variable_map (Dict[str, int]): A dictionary mapping variable names
                to their 1-based indices.
        """
        self._variable_map = variable_map
        self._transformer = AstTransformer(variable_map)
        self._grammar = self._build_grammar()

    def _build_grammar(self) -> pp.ParserElement:
        """
        Constructs the logical expression grammar using pyparsing objects.
        """
        variable = pp.Word(pp.alphas + "_", pp.alphanums + "_")
        variable.setParseAction(self._transformer.transform_variable)

        # --- Define grammar using infixNotation for precedence (highest to lowest) ---
        expr = infixNotation(
            variable,
            [
                ("!", 1, opAssoc.RIGHT, self._transformer.transform_unary_op),
                ("&&", 2, opAssoc.LEFT, self._transformer.transform_binary_op),
                ("||", 2, opAssoc.LEFT, self._transformer.transform_binary_op),
                ("^^", 2, opAssoc.LEFT, self._transformer.transform_binary_op),
                (pp.oneOf("=> <= = <=>"), 2, opAssoc.LEFT, self._transformer.transform_binary_op),
            ],
        )
        return expr

    def parse(self, rule_string: str) -> Any:
        """
        Parses a rule string and builds an Abstract Syntax Tree (AST).

        Args:
            rule_string (str): The logical rule to parse.

        Raises:
            ValueError: If the rule syntax is invalid or uses undefined variables.
            pyparsing.ParseException: For general parsing errors.

        Returns:
            Any: The Abstract Syntax Tree representing the rule.
        """
        if not rule_string:
            raise ValueError("Cannot parse an empty rule string.")

        try:
            ast = self._grammar.parseString(rule_string, parseAll=True)[0]
            return ast
        except (pp.ParseException, ValueError) as e:
            raise ValueError(f"Invalid rule syntax: {e}") from e
