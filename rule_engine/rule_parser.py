"""
High-performance rule string parsing module.

This module provides the `RuleParser` class, which leverages the `pyparsing`
library to efficiently parse logical rule strings into a standardized Abstract
Syntax Tree (AST). It defines the grammar for the rule language and handles
the transformation of the parsed output into a consistent, tuple-based AST
format used by the rest of the engine.
"""

from typing import Dict, Any

import pyparsing as pp
from pyparsing import infixNotation, opAssoc

# --- PERFORMANCE OPTIMIZATION ---
# Enable Packrat Caching globally for pyparsing. This significantly speeds up
# parsing of complex grammars by memoizing parsing results.
pp.ParserElement.enablePackrat()


class AstTransformer:
    """
    Transforms the pyparsing result into a standardized Abstract Syntax Tree (AST).

    The `pyparsing` library, by default, produces a nested list structure. This
    class provides a set of parse actions to convert this structure into a
    more explicit and consistent tuple-based format that the rule engine can
    easily process.

    Parameters
    ----------
    variable_map : Dict[str, int]
        A mapping from variable names to their internal 1-based integer indices.
        This is used to validate that variables in the rule string are defined.
    """

    def __init__(self, variable_map: Dict[str, int]):
        self._variable_map = variable_map

    def transform_variable(self, tokens: pp.ParseResults) -> Any:
        """
        Transform a variable token into a ('var', is_negated, name) tuple.

        Parameters
        ----------
        tokens : pp.ParseResults
            The parse results from pyparsing, containing the variable name.

        Returns
        -------
        Any
            A tuple representing the variable in the AST.

        Raises
        ------
        ValueError
            If the variable name is not found in the engine's `variable_map`.
        """
        var_name = tokens[0]
        if var_name not in self._variable_map:
            raise ValueError(f"Variable '{var_name}' is not defined in the engine.")
        return "var", False, var_name

    @staticmethod
    def transform_unary_op(tokens: pp.ParseResults) -> Any:
        """
        Transform a unary operation (negation) token.

        This handles the '!' operator. It directly negates a variable by flipping
        the `is_negated` flag in the variable's AST tuple. Negating a complex
        expression in parentheses is disallowed.

        Parameters
        ----------
        tokens : pp.ParseResults
            The parse results from pyparsing.

        Returns
        -------
        Any
            The transformed AST node for the negated variable.

        Raises
        ------
        ValueError
            If negation is applied to a parenthesized expression.
        """
        op, operand = tokens[0]
        if isinstance(operand, tuple) and operand[0] == "var":
            # Flip the negation flag of the variable
            return "var", not operand[1], operand[2]
        # It's a negated expression, which is not allowed.
        raise ValueError("Negation of expressions in parentheses is not allowed.")

    @staticmethod
    def transform_binary_op(tokens: pp.ParseResults) -> Any:
        """
        Transform a binary operation token into a nested AST structure.

        This handles operators like '&&', '||', '=>', etc. It constructs a
        left-associative tree of ('op', operator, left, right) tuples.

        Parameters
        ----------
        tokens : pp.ParseResults
            The parse results from pyparsing.

        Returns
        -------
        Any
            A nested tuple representing the binary operation in the AST.
        """
        tokens = tokens[0]
        node = tokens[0]
        for i in range(1, len(tokens), 2):
            op, right = tokens[i], tokens[i + 1]
            if op == "<=>":
                op = "="  # Standardize equivalence operator
            node = ("op", op, node, right)
        return node


class RuleParser:
    """
    Parses rule strings into an Abstract Syntax Tree (AST) using pyparsing.

    This class defines the grammar for logical expressions and uses the
    `AstTransformer` to convert parsed rule strings into a standardized AST
    format. The grammar supports standard logical operators with defined
    precedence.

    Parameters
    ----------
    variable_map : Dict[str, int]
        A dictionary mapping variable names to their 1-based integer indices.
        This is passed to the `AstTransformer` for variable validation.
    """

    def __init__(self, variable_map: Dict[str, int]):
        """
        Initialize the RuleParser and build the grammar.
        """
        self._variable_map = variable_map
        self._transformer = AstTransformer(variable_map)
        self._grammar = self._build_grammar()

    def _build_grammar(self) -> pp.ParserElement:
        """
        Construct the logical expression grammar using pyparsing objects.

        The grammar is defined using `infixNotation` to handle operator
        precedence correctly. The order of operators in the list defines
        their precedence from highest to lowest.

        Returns
        -------
        pp.ParserElement
            The complete, compiled parser element for the rule grammar.
        """
        # A variable is a standard Python identifier
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
        Parse a rule string and return its Abstract Syntax Tree (AST).

        Parameters
        ----------
        rule_string : str
            The logical rule string to parse (e.g., "a && !b => c").

        Returns
        -------
        Any
            The Abstract Syntax Tree representing the rule. The AST is a nested
            tuple structure. For example:
            - Variable: `('var', is_negated, 'name')`
            - Operation: `('op', '&&', left_child, right_child)`

        Raises
        ------
        ValueError
            If the rule string is empty, contains invalid syntax, or uses
            undefined variables.
        """
        if not rule_string:
            raise ValueError("Cannot parse an empty rule string.")

        try:
            # The [0] is to extract the single top-level match from the results
            ast = self._grammar.parseString(rule_string, parseAll=True)[0]
            return ast
        except (pp.ParseException, ValueError) as e:
            # Catch both pyparsing errors and our custom ValueErrors from transformers
            raise ValueError(f"Invalid rule syntax: {e}") from e
