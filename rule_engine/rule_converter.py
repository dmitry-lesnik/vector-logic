"""
This module defines the RuleConverter class, which handles the end-to-end
process of converting a rule string into a StateVector.
"""

from typing import Dict, Any, List, Tuple

from .rule_parser import RuleParser
from .state_vector import StateVector
from .t_object import TObject

# Type alias for the Abstract Syntax Tree (AST) node structure for clarity.
# An AST node is a tuple, e.g., ("op", "&&", left_node, right_node).
ASTNode = Tuple[Any, ...]


class RuleConverter:
    """
    Orchestrates the conversion of a rule string into a final, combined StateVector.

    This class acts as a facade, handling a multi-step process:
    1. Parsing the rule string into an Abstract Syntax Tree (AST).
    2. Traversing the AST to replace repeated variables with dummies and generate
       corresponding equality rules.
    3. Flattening the modified AST into a list of simple, solvable ASTs.
    4. Converting each simple AST into a StateVector.
    5. Multiplying all StateVectors together.
    6. Removing all temporary auxiliary/dummy variables from the final StateVector.
    """

    def __init__(self, variable_map: Dict[str, int]):
        """
        Initializes the converter with the engine's variable-to-index mapping.

        Args:
            variable_map: A dictionary mapping variable names to their indices.
        """
        self._variable_map = variable_map
        # The parser is now an internal, stateless tool.
        self._parser = RuleParser(self._variable_map)
        # Internal state is reset for each conversion.
        self._aux_var_counter = 0
        self._aux_var_map: Dict[str, int] = {}

        # for debugging
        self._all_simple_asts = None

    def convert(self, rule_string: str) -> StateVector:
        """
        Converts a rule string into a final StateVector, fully handling and
        removing any intermediate auxiliary variables.

        Returns:
            StateVector: The final, combined StateVector for the entire rule.
        """
        # Reset internal state for each conversion run.
        self._aux_var_counter = 0
        self._aux_var_map = {}

        # 1. Parse the original rule string into an AST.
        ast = self._parser.parse(rule_string)

        # 2. Handle repeated variables by traversing and modifying the AST.
        modified_ast, equality_asts = self._handle_repeated_variables_in_ast(ast)

        # 3. Flatten the main AST and combine with equality rules.
        flattened_asts = self._flatten(modified_ast)
        self._all_simple_asts = flattened_asts + equality_asts

        # The full map includes variables from flattening and preprocessing.
        full_variable_map = self._variable_map.copy()
        full_variable_map.update(self._aux_var_map)

        # 4. Convert each simple AST rule into a StateVector.
        state_vectors = [self._visit(simple_ast, full_variable_map) for simple_ast in self._all_simple_asts]

        # 5. Multiply all the simple state vectors together.
        final_sv = StateVector([TObject()]) if not state_vectors else state_vectors[0]
        for sv in state_vectors[1:]:
            final_sv *= sv

        # 6. Remove all temporary auxiliary variables.
        if self._aux_var_map:
            aux_indices = list(self._aux_var_map.values())
            final_sv = final_sv.remove_variables(aux_indices).simplify(max_num_iter=None, reduce_subsumption=True)

        return final_sv

    def _handle_repeated_variables_in_ast(self, ast: ASTNode) -> Tuple[ASTNode, List[ASTNode]]:
        """
        Traverses an AST, replacing duplicate variable occurrences with dummies
        and generating equality constraint ASTs.
        """
        seen_vars: Dict[str, int] = {}
        equality_asts: List[ASTNode] = []
        modified_ast = self._replace_duplicates_recursive(ast, seen_vars, equality_asts)
        return modified_ast, equality_asts

    def _replace_duplicates_recursive(
        self, node: ASTNode, seen_vars: Dict[str, int], equality_asts: List[ASTNode]
    ) -> ASTNode:
        """
        Recursively rebuilds the AST, replacing duplicate variables.
        """
        node_type = node[0]
        if node_type == "var":
            is_negated, var_name = node[1], node[2]
            if var_name not in seen_vars:
                # First occurrence of this variable.
                seen_vars[var_name] = 1
                return node
            else:
                # This is a repeated variable, create a dummy.
                self._aux_var_counter += 1
                dummy_name = f"__aux_{self._aux_var_counter}"
                self._aux_var_map[dummy_name] = -self._aux_var_counter

                # Create the equality rule AST: original_var = dummy_var
                equality_ast: ASTNode = (
                    "op",
                    "=",
                    ("var", False, var_name),
                    ("var", False, dummy_name),
                )
                equality_asts.append(equality_ast)

                # Return a new 'var' node for the dummy to replace this one,
                # preserving the original negation status.
                return "var", is_negated, dummy_name

        if node_type == "op":
            # Recurse on children and rebuild the operator node.
            op, left, right = node[1], node[2], node[3]
            new_left = self._replace_duplicates_recursive(left, seen_vars, equality_asts)
            new_right = self._replace_duplicates_recursive(right, seen_vars, equality_asts)
            return "op", op, new_left, new_right

        # Should not be reached with the current grammar.
        raise TypeError(f"Unexpected AST node structure: {node}")

    def _flatten(self, ast: ASTNode) -> List[ASTNode]:
        """Decomposes a complex AST into a list of simple ASTs."""
        if ast[0] == "var":
            return [ast]

        # Check if the AST is already a simple triplet or binary rule.
        if ast[0] == "op":
            _, op, left, right = ast
            if left[0] == "var" and right[0] == "var":
                return [ast]
            if op == "=":
                if left[0] == "var" and right[0] == "op":
                    _, _, inner_left, inner_right = right
                    if inner_left[0] == "var" and inner_right[0] == "var":
                        return [ast]
                if left[0] == "op" and right[0] == "var":
                    # This case should be handled by grammar (e.g. left-associativity)
                    # but check for safety.
                    _, _, inner_left, inner_right = left
                    if inner_left[0] == "var" and inner_right[0] == "var":
                        return [ast]

        simple_asts: List[ASTNode] = []
        final_rule = self._flatten_recursive(ast, simple_asts, is_root=True)
        simple_asts.append(final_rule)
        return simple_asts

    def _flatten_recursive(self, node: ASTNode, simple_asts: List[ASTNode], is_root: bool) -> ASTNode:
        """Recursive helper for flattening the AST."""
        if node[0] == "var":
            return node

        op, left, right = node[1], node[2], node[3]
        left_repr = self._flatten_recursive(left, simple_asts, is_root=False)
        right_repr = self._flatten_recursive(right, simple_asts, is_root=False)
        current_rule: ASTNode = ("op", op, left_repr, right_repr)

        if is_root:
            return current_rule

        # This is an intermediate node, so create an auxiliary variable.
        self._aux_var_counter += 1
        aux_var_name = f"__aux_{self._aux_var_counter}"
        self._aux_var_map[aux_var_name] = -self._aux_var_counter
        aux_var_node: ASTNode = ("var", False, aux_var_name)
        equivalence_rule: ASTNode = ("op", "=", aux_var_node, current_rule)
        simple_asts.append(equivalence_rule)

        return aux_var_node

    def _visit(self, node: ASTNode, var_map: Dict[str, int]) -> StateVector:
        """Visits a simple AST node to build its StateVector."""
        node_type = node[0]
        if node_type == "var":
            return self._visit_var(node, var_map)
        if node_type == "op":
            return self._visit_op(node, var_map)
        raise ValueError(f"Unknown AST node type: {node_type}")

    @staticmethod
    def _visit_var(node: ASTNode, var_map: Dict[str, int]) -> StateVector:
        """Handles a simple variable node."""
        is_negated, var_name = node[1], node[2]
        var_index = var_map[var_name]
        t_obj = TObject(zeros={var_index}) if is_negated else TObject(ones={var_index})
        return StateVector([t_obj])

    @staticmethod
    def _create_triplet_sv(op: str, idx1: int, idx2: int, idx3: int) -> StateVector:
        """Creates a StateVector for a triplet rule: x1 = (x2 op x3)."""
        op_map = {
            "&&": [
                TObject(ones={idx1, idx2, idx3}),
                TObject(zeros={idx1, idx2}),
                TObject(ones={idx2}, zeros={idx1, idx3}),
            ],
            "||": [
                TObject(ones={idx1, idx2}),
                TObject(ones={idx1, idx3}, zeros={idx2}),
                TObject(zeros={idx1, idx2, idx3}),
            ],
            "^^": [
                TObject(ones={idx1, idx2}, zeros={idx3}),
                TObject(ones={idx1, idx3}, zeros={idx2}),
                TObject(zeros={idx1, idx2, idx3}),
                TObject(ones={idx2, idx3}, zeros={idx1}),
            ],
            "=>": [
                TObject(ones={idx1, idx2, idx3}),
                TObject(ones={idx1}, zeros={idx2}),
                TObject(ones={idx2}, zeros={idx1, idx3}),
            ],
            "<=": [
                TObject(ones={idx1, idx2}),
                TObject(ones={idx1}, zeros={idx2, idx3}),
                TObject(ones={idx3}, zeros={idx1, idx2}),
            ],
            "=": [
                TObject(ones={idx1, idx2, idx3}),
                TObject(ones={idx1}, zeros={idx2, idx3}),
                TObject(ones={idx2}, zeros={idx1, idx3}),
                TObject(ones={idx3}, zeros={idx1, idx2}),
            ],
        }
        t_objs = op_map.get(op)
        if t_objs is None:
            raise NotImplementedError(f"Triplet operator for '{op}' not implemented.")
        return StateVector(t_objects=t_objs)

    def _visit_op(self, node: ASTNode, var_map: Dict[str, int]) -> StateVector:
        """Handles a binary operation node."""
        op, left, right = node[1], node[2], node[3]
        left_is_var, right_is_var = left[0] == "var", right[0] == "var"

        # Case: Simple binary rule like x1 op x2
        if left_is_var and right_is_var:
            l_neg, l_name = left[1], left[2]
            r_neg, r_name = right[1], right[2]
            idx1, idx2 = var_map[l_name], var_map[r_name]
            vars_to_negate = [idx for idx, is_neg in [(idx1, l_neg), (idx2, r_neg)] if is_neg]

            op_map = {
                "&&": StateVector([TObject(ones={idx1, idx2})]),
                "||": StateVector([TObject(ones={idx1}), TObject(ones={idx2}, zeros={idx1})]),
                "^^": StateVector([TObject(ones={idx1}, zeros={idx2}), TObject(ones={idx2}, zeros={idx1})]),
                "=>": StateVector([TObject(ones={idx1, idx2}), TObject(zeros={idx1})]),
                "<=": StateVector([TObject(ones={idx1}), TObject(zeros={idx1, idx2})]),
                "=": StateVector([TObject(ones={idx1, idx2}), TObject(zeros={idx1, idx2})]),
            }
            sv = op_map.get(op)
            if sv is None:
                raise NotImplementedError(f"Binary operator '{op}' not implemented.")
            return sv.negate_variables(vars_to_negate)

        # Case: Simple triplet rule like x1 = (x2 op x3)
        if op == "=":
            triplet, single = (right, left) if right[0] == "op" else (left, right)
            if single[0] == "var" and triplet[0] == "op":
                _, inner_op, i_left, i_right = triplet
                if i_left[0] == "var" and i_right[0] == "var":
                    s_neg, s_name = single[1], single[2]
                    l_neg, l_name = i_left[1], i_left[2]
                    r_neg, r_name = i_right[1], i_right[2]
                    idx1, idx2, idx3 = var_map[s_name], var_map[l_name], var_map[r_name]
                    # The indices must be negated based on their state in the AST
                    # before being passed to the triplet creator.
                    vars_to_negate = [idx for idx, is_neg in [(idx1, s_neg), (idx2, l_neg), (idx3, r_neg)] if is_neg]
                    # Create the base StateVector for the triplet.
                    sv = self._create_triplet_sv(inner_op, idx1, idx2, idx3)
                    # Apply negations to get the final correct vector.
                    return sv.negate_variables(vars_to_negate)

        raise NotImplementedError(f"Unsupported AST structure for direct conversion: {node}")
