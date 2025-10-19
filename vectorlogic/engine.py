"""
Core module for the rule engine.

This module provides the main `Engine` class, which orchestrates the entire
process of rule management, compilation, and inference. It also defines
the `InferenceResult` class for handling the outcomes of predictions.
"""

import re
from typing import Dict, List, Optional, Tuple

from . import helpers
from .rule_converter import RuleConverter
from .state_vector import StateVector
from .t_object import TObject


class InferenceResult:
    """
    A wrapper for the result of a prediction.

    This class provides a user-friendly way to query the state of variables
    by name from a resulting StateVector, without needing to know about
    internal variable indices.

    Parameters
    ----------
    state_vector : StateVector
        The StateVector that this result wraps.
    variable_map : Dict[str, int]
        A mapping from variable names to their internal integer indices.
    """

    def __init__(self, state_vector: StateVector, variable_map: Dict[str, int]):
        self._state_vector = state_vector
        self._variable_map = variable_map

    @property
    def state_vector(self) -> StateVector:
        """The raw StateVector result of the inference."""
        return self._state_vector

    def size(self) -> int:
        """
        Return the size of the underlying StateVector.
        The size represents the number of TObjects in the StateVector.

        Returns
        -------
        int
            The number of TObjects.
        """
        return self._state_vector.size()

    def get_value(self, variable_name: str) -> Optional[int]:
        """
        Gets the consolidated value of a variable from this inference result.

        Parameters
        ----------
        variable_name : str
            The name of the variable to query.

        Returns
        -------
        Optional[int]
            - 1 if the variable is determined to be True.
            - 0 if the variable is determined to be False.
            - -1 if the variable's state is undetermined or mixed.
            - None if the underlying StateVector is a contradiction.

        Raises
        ------
        ValueError
            If `variable_name` was not part of the engine's context.
        """
        if variable_name not in self._variable_map:
            raise ValueError(f"Variable '{variable_name}' was not part of the engine's context.")

        if self.is_contradiction():
            return None

        index = self._variable_map[variable_name]
        return self._state_vector.var_value(index)

    def is_contradiction(self) -> bool:
        """
        Checks if this inference result is a contradiction.

        An inference result is a contradiction if its underlying StateVector
        has no valid states.

        Returns
        -------
        bool
            True if the result is a contradiction, False otherwise.
        """
        return self._state_vector.is_contradiction()

    def __bool__(self) -> bool:
        """
        Converts the InferenceResult to a boolean.

        An InferenceResult is considered "truthy" if it is not a contradiction,
        and "falsy" if it is a contradiction. This allows for checks like:
        `if result:` (true if not a contradiction) or
        `if not result:` (true if it is a contradiction).

        Returns
        -------
        bool
            False if the result is a contradiction, True otherwise.
        """
        return not self.is_contradiction()

    def print(self, max_index: Optional[int] = None, indent: int = 0):
        """
        Prints the string representation of the underlying StateVector.

        Parameters
        ----------
        max_index : int, optional
            The largest index to display for alignment. If None, it is
            auto-calculated. Defaults to None.
        indent : int, optional
            The number of spaces to indent the output. Defaults to 0.
        """
        self._state_vector.print(max_index=max_index, indent=indent)


class Engine:
    """
    The main class for the rule engine.

    This class manages the lifecycle of a rule-based system, including
    variable definitions, rule addition, compilation of the knowledge base,
    and performing inference.

    Parameters
    ----------
    variables : List[str]
        A list of all variable names to be used in the engine.
    name : str, optional
        An optional name for the engine instance for identification.
        Defaults to None.
    rules : List[str], optional
        An optional list of initial rule strings to add upon initialization.
        Defaults to None.

    Attributes
    ----------
    _variables : List[str]
        A sorted, unique list of variable names.
    _variable_map : Dict[str, int]
        A mapping from variable names to their 1-based integer indices.
    _uncompiled_rules : List[str]
        A list of rules that have been added but not yet compiled.
    _state_vectors : List[StateVector]
        The list of StateVectors corresponding to uncompiled rules.
    _valid_set : Optional[StateVector]
        The compiled StateVector representing the entire knowledge base.
        Is None until `compile()` is called.
    _is_compiled : bool
        A flag indicating whether the engine has a compiled valid set.
    """

    def __init__(
        self,
        variables: List[str],
        name: Optional[str] = None,
        rules: Optional[List[str]] = None,
    ):
        self._validate_variables(variables)
        self._variables: List[str] = sorted(list(set(variables)))
        self._name: Optional[str] = name
        self._variable_map: Dict[str, int] = {var: i + 1 for i, var in enumerate(self._variables)}

        # --- Core State ---
        self._uncompiled_rules: List[str] = []
        self._state_vectors: List[StateVector] = []
        self._valid_set: Optional[StateVector] = None
        self._is_compiled: bool = False

        # --- Debugging & History ---
        self._compiled_rules: List[str] = []
        self._intermediate_sizes: List[int] = []

        for rule in rules or []:
            self.add_rule(rule)

    @property
    def variables(self) -> List[str]:
        """Return the sorted list of unique variable names."""
        return self._variables

    @property
    def rules(self) -> List[str]:
        """The list of uncompiled rule strings in the engine."""
        return self._uncompiled_rules

    @property
    def state_vectors(self) -> List[StateVector]:
        """The list of uncompiled state vectors in the engine."""
        return self._state_vectors

    @property
    def compiled(self) -> bool:
        """True if the engine has been compiled, False otherwise."""
        return self._is_compiled

    @property
    def intermediate_sizes(self) -> List[int]:
        """
        Return sizes of intermediate state vectors during compilation.

        This is a debugging tool to inspect the complexity of the compilation process.

        Returns
        -------
        List[int]
            A list of intermediate StateVector sizes.
        """
        return self._intermediate_sizes

    @property
    def valid_set(self) -> StateVector:
        """
        The compiled knowledge base ('valid set') of the engine.

        .. warning::
           This method will raise an `AttributeError` if the engine has not
           been compiled.

        Returns
        -------
        StateVector
            The compiled StateVector representing the knowledge base.

        Raises
        ------
        AttributeError
            If the engine has not been compiled yet. Call `.compile()` first
            to generate the valid set.
        """
        if not self._is_compiled or self._valid_set is None:
            raise AttributeError("The 'valid_set' is not available. Call .compile() to build it.")
        return self._valid_set

    def add_rule(self, rule_string: str):
        """
        Adds and converts a new rule to the engine's uncompiled set.

        This method parses a string representation of a logical rule and
        converts it into its corresponding StateVector representation. The
        new rule is added to a pending list, ready for compilation. Adding a
        new rule will mark the engine as "not compiled".

        Parameters
        ----------
        rule_string : str
            The logical rule to add.

        Notes
        -----
        The rule syntax supports standard propositional logic operators.
        Variables must be valid Python identifiers (letters, numbers,
        underscores) and must have been declared when the Engine was
        initialized.

        Supported Operators (in order of precedence):
        - `!`         : NOT (Negation)
        - `&&`        : AND
        - `||`        : OR
        - `^^`        : XOR (Exclusive OR)
        - `=>`        : IMPLIES
        - `<=`        : IS IMPLIED BY
        - `=` / `<=>` : EQUIVALENT

        Use parentheses `()` to group expressions and override default
        operator precedence.

        Examples
        --------
        >>> engine.add_rule("sky_is_grey && humidity_is_high => it_will_rain")
        >>> engine.add_rule("take_umbrella = (it_will_rain || has_forecast)")
        >>> engine.add_rule("!wind_is_strong")
        """
        self._uncompiled_rules.append(rule_string)
        converter = RuleConverter(self._variable_map)
        state_vector = converter.convert(rule_string)
        self._state_vectors.append(state_vector)
        self._is_compiled = False

    def add_evidence(self, evidence: Dict[str, bool]):
        """
        Adds an evidence statement to the engine's uncompiled set.

        Parameters
        ----------
        evidence : Dict[str, bool]
            A dictionary of variable names and their boolean values.
        """
        ones = {self._variable_map[var] for var, val in evidence.items() if val}
        zeros = {self._variable_map[var] for var, val in evidence.items() if not val}
        t_object = TObject(ones=ones, zeros=zeros)
        state_vector = StateVector([t_object])

        self._uncompiled_rules.append(f"evidence: {evidence}")
        self._state_vectors.append(state_vector)
        self._is_compiled = False

    def add_state_vector(self, state_vector: StateVector):
        """
        Adds a custom StateVector to the engine's uncompiled set.

        Parameters
        ----------
        state_vector : StateVector
            A StateVector to add.
        """
        self._uncompiled_rules.append("custom state vector")
        self._state_vectors.append(state_vector)
        self._is_compiled = False

    def compile(self):
        """
        Compiles all uncompiled rules into the engine's 'valid set'.

        This method takes all pending `StateVector`s, multiplies them
        with the existing `_valid_set` (if any), and stores the final result.
        The list of uncompiled vectors is then cleared. This is an explicit,
        user-driven action.
        """
        if self._is_compiled:
            return

        all_svs = self._state_vectors
        if self._valid_set is not None:
            all_svs.append(self._valid_set)

        def _finalize_compilation():
            self._is_compiled = True
            self._compiled_rules.extend(self._uncompiled_rules)
            self._uncompiled_rules.clear()
            self._state_vectors.clear()

        if not all_svs:
            self._valid_set = StateVector([TObject()])
            self._intermediate_sizes.append(self._valid_set.size())
            _finalize_compilation()
            return

        valid_set, int_sizes = self.multiply_all_vectors(all_svs)
        self._valid_set = valid_set.simplify()
        self._intermediate_sizes.extend(int_sizes)
        _finalize_compilation()

    def predict(self, evidence: Dict[str, bool]) -> InferenceResult:
        """
        Calculates an inference result without altering the engine's state.

        This method performs an on-the-fly calculation by multiplying all
        StateVectors in the engine (both compiled and uncompiled) with a new
        StateVector generated from the provided evidence.

        Parameters
        ----------
        evidence : Dict[str, bool]
            A dictionary of variable names and their boolean values.

        Returns
        -------
        InferenceResult
            A result object wrapping the final StateVector from this inference.
        """
        ones = {self._variable_map[var] for var, val in evidence.items() if val}
        zeros = {self._variable_map[var] for var, val in evidence.items() if not val}
        evidence_sv = StateVector([TObject(ones=ones, zeros=zeros)])

        all_svs = self._state_vectors.copy()
        if self._valid_set is not None:
            all_svs.append(self._valid_set)
        all_svs.append(evidence_sv)

        result_sv, int_sizes = self.multiply_all_vectors(all_svs)
        self._intermediate_sizes = int_sizes
        return InferenceResult(result_sv, self._variable_map)

    def get_variable_value(self, variable_name: str) -> Optional[int]:
        """
        Gets a variable's value from the compiled 'valid set'.

        .. warning::
           This method will raise an `AttributeError` if the engine has not
           been compiled.

        Parameters
        ----------
        variable_name : str
            The name of the variable to query.

        Returns
        -------
        Optional[int]
            - 1 if the variable is True.
            - 0 if the variable is False.
            - -1 if the variable is undetermined.
            - None if the valid set is a contradiction.

        Raises
        ------
        ValueError
            If `variable_name` is not defined in the engine.
        AttributeError
            If the engine is not compiled.
        """
        if variable_name not in self._variable_map:
            raise ValueError(f"Variable '{variable_name}' not defined in the engine.")

        # Accessing the property will raise an AttributeError if not compiled
        valid_set = self.valid_set

        if valid_set.is_contradiction():
            return None

        return valid_set.var_value(self._variable_map[variable_name])

    def is_contradiction(self) -> bool:
        """
        Checks if the compiled 'valid set' is a contradiction.

        .. warning::
           This method will raise an `AttributeError` if the engine has not
           been compiled.

        Returns
        -------
        bool
            True if the valid set is a contradiction, False if it is not.

        Raises
        ------
        AttributeError
            If the engine is not compiled.
        """
        # Accessing the property will raise an AttributeError if not compiled
        return self.valid_set.is_contradiction()

    def print(self, debug_info: bool = False):
        """
        Prints a formatted summary of the engine's state.

        Parameters
        ----------
        debug_info : bool, optional
            If True, prints additional debugging information, including the
            history of compiled rules and the evolution of StateVector sizes
            during compilation. Defaults to False.
        """
        if self._name:
            print(f"====== Engine: {self._name} ======")
        else:
            print("====== Engine ======")

        print("\n✅ Engine Compiled" if self._is_compiled else "\n🟨 Not Compiled")
        print(f"\nVariables: {self._variables}")
        inverse_map = {v: k for k, v in self._variable_map.items()}
        print(f"Variable Map: {inverse_map}")
        max_index = len(self._variables)

        if debug_info:
            print(f"\n--- Compiled Rules [{len(self._compiled_rules)}] ---")
            for i, rule in enumerate(self._compiled_rules):
                print(f"{i + 1}.  {rule}")

        print(f"\n--- Uncompiled Rules [{len(self._uncompiled_rules)}] ---")
        for i, rule in enumerate(self._uncompiled_rules):
            print(f"\n{i + 1}. Rule:  {rule}")
            self._state_vectors[i].print(max_index=max_index, indent=4, print_brackets=False)

        print("\n--- Compiled Valid Set ---" if self._is_compiled else "\n--- Valid Set (Not yet compiled) ---")
        if self._is_compiled and self._valid_set:
            self._valid_set.print(max_index=max_index, indent=4, print_brackets=False)
        else:
            print("    (Empty)")

        if debug_info:
            print("\n--- State Vector sizes evolution during compilation:")
            print(f"    {self._intermediate_sizes}")

        print("\n==============================")

    @staticmethod
    def _validate_variables(variables: List[str]):
        """
        Validate variable names.

        Checks if all variable names are "conformal". A variable name is considered
        conformal if it contains only alphanumeric characters and underscores, and
        does not start with a number.

        Parameters
        ----------
        variables : List[str]
            The list of variable names to validate.

        Raises
        ------
        ValueError
            If a variable name is not conformal.
        """
        conformal_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
        for var in variables:
            if not conformal_pattern.match(var):
                raise ValueError(f"Variable name '{var}' is not conformal.")

    @staticmethod
    def multiply_all_vectors(
        state_vectors: List[StateVector], max_cluster_size: int = 2
    ) -> Tuple[StateVector, List[int]]:
        """
        Multiplies a list of StateVectors using an optimized clustering strategy.

        Parameters
        ----------
        state_vectors : List[StateVector]
            The list of StateVectors to multiply.
        max_cluster_size : int, optional
            The maximum number of vectors to group in a single multiplication
            step. Defaults to 2.

        Returns
        -------
        Tuple[StateVector, List[int]]
            - The final product StateVector.
            - A list of intermediate StateVector sizes for debugging.

        Notes
        -----
        This method uses a heuristic to decide the order of multiplication.
        It iteratively finds the cluster of vectors with the most similar
        variable usage (highest Jaccard similarity between their pivot sets)
        and multiplies them first. This can significantly reduce the size of
        intermediate StateVectors and speed up compilation.
        """
        # --- Handle simple cases and perform initial cleanup ---
        if len(state_vectors) == 0:
            return StateVector(), [0]
        remaining_svs = []
        for sv in state_vectors:
            if sv.is_contradiction():
                return StateVector(), [0]  # Early exit
            if not sv.is_trivial():
                remaining_svs.append(sv)

        if not remaining_svs:
            return StateVector([TObject()]), [1]  # All were trivial
        if len(remaining_svs) == 1:
            return remaining_svs[0], [remaining_svs[0].size()]
        if len(remaining_svs) == 2:
            product_sv = remaining_svs[0] * remaining_svs[1]
            return product_sv, [product_sv.size()]

        # --- Main optimized multiplication loop ---
        intermediate_sizes = []
        pivot_sets = [sv.pivot_set() for sv in remaining_svs]
        union_sizes, intersection_sizes = helpers.calc_ps_unions_intersections(pivot_sets)

        while len(remaining_svs) > 1:
            cluster_indices = helpers.find_next_cluster(pivot_sets, union_sizes, intersection_sizes, max_cluster_size)

            # Multiply the vectors in the chosen cluster
            product_sv = remaining_svs[cluster_indices[0]]
            for i in cluster_indices[1:]:
                product_sv *= remaining_svs[i]

                intermediate_sizes.append(product_sv.size())
                if product_sv.is_contradiction():
                    return StateVector(), intermediate_sizes

            # --- Update state for the next iteration ---
            # Remove the original vectors from the list (in reverse order) and matrices
            sorted_indices = sorted(cluster_indices, reverse=True)
            for i in sorted_indices:
                del remaining_svs[i]
                del pivot_sets[i]

            # Add the new product back for the next round
            remaining_svs.append(product_sv)
            pivot_sets.append(product_sv.pivot_set())

            # Update similarity matrices efficiently if there's more work to do
            if len(remaining_svs) > 1:
                union_sizes, intersection_sizes = helpers.update_ps_unions_intersections(
                    union_sizes, intersection_sizes, sorted_indices, pivot_sets
                )

        return remaining_svs[0], intermediate_sizes
