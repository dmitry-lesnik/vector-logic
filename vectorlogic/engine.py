"""
Core module for the rules engine.

This module provides the main `Engine` class, which orchestrates the entire
process of rule management, compilation, and inference. It also defines
the `InferenceResult` class for handling the outcomes of predictions.
"""

import re
from typing import Dict, List, Optional, Tuple, Union, Iterable

import numpy as np

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
        self._index_to_name = {v: k for k, v in self._variable_map.items()}

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

    def iter_dicts(self) -> Iterable[Optional[Dict[str, bool]]]:
        """
        Yield a dictionary for each TObject in the StateVector.

        This method provides an iterator that converts each TObject into a
        dictionary of variable names and boolean values.

        Yields
        -------
        Iterable[Optional[Dict[str, bool]]]
            An iterator of dictionaries, where each dictionary represents a TObject.
        """
        for t_obj in self._state_vector:
            yield t_obj.to_dict(self._index_to_name)


class Engine:
    """
    The main class for the rules engine.

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
    verbose : int, optional
        Verbosity level (0 for silent). Defaults to 0.

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
        verbose: int = 0,
    ):
        self._validate_variables(variables)
        self._variables: List[str] = sorted(list(set(variables)))
        self._name: Optional[str] = name
        self._variable_map: Dict[str, int] = {var: i + 1 for i, var in enumerate(self._variables)}
        self._index_to_name: Dict[int, str] = {v: k for k, v in self._variable_map.items()}
        self._verbose = verbose

        if self._verbose > 0:
            engine_name = f"'{self._name}' " if self._name else ""
            print(f"Engine {engine_name}initialized with {len(self._variables)} variables.")

        # --- Core State ---
        self._uncompiled_rules: List[str] = []
        self._state_vectors: List[StateVector] = []
        self._valid_set: Optional[StateVector] = None
        self._is_compiled: bool = False

        # --- Debugging & History ---
        self._compiled_rules: List[str] = []
        self._intermediate_sizes: List[int] = []

        # --- Optimisation Hyper-parameters ---
        # (Advanced users can tune these on the instance)
        self._opt_predator_base: float = 0.6
        self._opt_predator_threshold: float = 1.2
        self._opt_max_predator_size: int = 2
        self._opt_max_cluster_size: int = 2
        # --- End ---

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
    def intermediate_sizes_stats(self) -> Dict[str, float]:
        """
        Return statistics about the sizes of intermediate state vectors during compilation.

        This is a debugging tool to inspect the complexity of the compilation process.

        Returns
        -------
        Dict[str, float]
            A dictionary containing 'num_entries', 'min', 'mean', 'rms', and 'max'
            statistics of the intermediate StateVector sizes.
        """
        if not self._intermediate_sizes:
            return {
                "num_entries": 0,
                "min": np.nan,
                "mean": np.nan,
                "rms": np.nan,
                "max": np.nan,
                "last": np.nan,
            }

        sizes_array = np.array(self._intermediate_sizes)
        return {
            "num_entries": len(self._intermediate_sizes),
            "min": int(np.min(sizes_array)),
            "mean": float(np.round(np.mean(sizes_array), 1)),
            "rms": float(np.round(np.sqrt(np.mean(sizes_array**2)), 1)),
            "max": int(np.max(sizes_array)),
            "last": int(self._intermediate_sizes[-1]),
        }

    @property
    def opt_config(self) -> Dict[str, Union[float, int]]:
        """Returns a dictionary of the current optimisation hyper-parameters."""
        return {
            "predator_base": self._opt_predator_base,
            "predator_threshold": self._opt_predator_threshold,
            "max_predator_size": self._opt_max_predator_size,
            "max_cluster_size": self._opt_max_cluster_size,
        }

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

    def valid_set_iter_dicts(self):
        """
        Iterates through the valid rows of the compiled knowledge base.

        Each t-object (a row in the valid set) is yielded as a dictionary
        mapping variable names to their boolean values.

        .. warning::
           This method will raise an `AttributeError` if the engine has not
           been compiled.

        Yields
        ------
        Dict[str, bool]
            A dictionary representing a single valid state.

        Raises
        ------
        AttributeError
            If the engine has not been compiled yet. Call `.compile()` first.
        """
        # Accessing self.valid_set will automatically check for compilation
        # and raise an AttributeError if not compiled.
        for t_obj in self.valid_set:
            yield t_obj.to_dict(self._index_to_name)

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
        if self._verbose > 0:
            print(f'Adding rule: "{rule_string}"')
        self._uncompiled_rules.append(rule_string)
        converter = RuleConverter(self._variable_map)
        state_vector = converter.convert(rule_string)
        self._state_vectors.append(state_vector)
        self._is_compiled = False

    def add_evidence(self, evidence: Dict[str, bool]):
        """
        Adds an evidence statement as a new rule to the uncompiled set.

        This method provides a convenient way to assert that a set of variables
        have specific boolean values. It is logically equivalent to adding a
        rule that is a conjunction of literals. For example, the
        evidence `{x1: True, x2: False}` is equivalent to
        adding the rule `(x1 && !x2)`.

        Adding new evidence marks the engine as "not compiled".

        Parameters
        ----------
        evidence : Dict[str, bool]
            A dictionary of variable names and their boolean values.
        """
        if self._verbose > 0:
            print(f"Adding evidence: {evidence}")
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
        if self._verbose > 0:
            print(f"Adding custom StateVector of size {state_vector.size()}")
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
            if self._verbose > 0:
                print("Engine.compile(): Engine already compiled.")
            return

        all_svs = self._state_vectors
        if self._valid_set is not None:
            all_svs.append(self._valid_set)

        if self._verbose > 0:
            print(f"Engine.compile(): Compiling {len(all_svs)} state vectors...")

        def _finalize_compilation():
            self._is_compiled = True
            self._compiled_rules.extend(self._uncompiled_rules)
            self._uncompiled_rules.clear()
            self._state_vectors.clear()
            if self._verbose > 0:
                print(f"Engine.compile(): Compilation finished. Final size = {self._valid_set.size()}")

        if not all_svs:
            self._valid_set = StateVector([TObject()])
            self._intermediate_sizes.append(self._valid_set.size())
            _finalize_compilation()
            return

        valid_set, int_sizes = self.multiply_all_vectors(all_svs, self.opt_config, verbose=self._verbose)
        self._valid_set = valid_set.simplify()
        self._intermediate_sizes.extend(int_sizes)
        _finalize_compilation()

    def predict(self, evidence: Dict[str, bool]) -> InferenceResult:
        """
        Calculates an inference result without altering the engine's state.

        This method performs an on-the-fly inference by temporarily combining
        the provided evidence with the engine's existing knowledge base (both
        compiled and uncompiled rules). The evidence is not permanently added
        to the engine.

        Parameters
        ----------
        evidence : Dict[str, bool]
            A dictionary of variable names and their boolean values.

        Returns
        -------
        InferenceResult
            A result object wrapping the final StateVector from this inference.

        Notes
        -----
        The evidence dictionary is treated as a temporary rule for this
        inference. It is logically equivalent to a conjunction of literals.
        For example, providing the evidence `{x1: True, x2: False}` is
        the same as temporarily adding the rule `x1 && !x2` for this
        calculation.
        """
        ones = {self._variable_map[var] for var, val in evidence.items() if val}
        zeros = {self._variable_map[var] for var, val in evidence.items() if not val}
        evidence_sv = StateVector([TObject(ones=ones, zeros=zeros)])

        all_svs = self._state_vectors.copy()
        if self._valid_set is not None:
            all_svs.append(self._valid_set)
        all_svs.append(evidence_sv)

        if self._verbose == 1:
            print(f"Engine.predict(): Num state vectors = {len(all_svs)}")
        elif self._verbose > 1:
            print(f"Engine.predict(): Num state vectors = {len(all_svs)}, Evidence: {evidence}")

        result_sv, int_sizes = self.multiply_all_vectors(all_svs, self.opt_config, verbose=self._verbose)
        if self._verbose > 0:
            print(f"Engine.predict(): Multiplication finished. Final state vector size = {result_sv.size()}")

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

        if self._verbose >= 2 or debug_info:
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

        if self._verbose >= 2 or debug_info:
            print("\n--- State Vector sizes evolution during compilation:")
            if self._intermediate_sizes:
                print(f"    {self.intermediate_sizes_stats}")
            else:
                print("    No intermediate sizes recorded.")

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
    def _update_multiplication_state(
        remaining_svs: List[StateVector],
        pivot_sets: List[set],
        sv_sizes: List[int],
        union_sizes: np.ndarray,
        intersection_sizes: np.ndarray,
        indices_to_remove: List[int],
        new_products: List[StateVector],
    ) -> Tuple[bool, List[StateVector], List[set], List[int], np.ndarray, np.ndarray]:
        """
        Helper to update the state lists after a multiplication step.

        Modifies the lists in place by deleting old items and extending with
        new ones.

        Returns:
            Tuple: (is_finished, updated_svs, updated_pivots, updated_sizes,
                    updated_unions, updated_intersections)
        """
        # ---- Update the lists --------
        sorted_indices = sorted(list(indices_to_remove), reverse=True)
        for i in sorted_indices:
            del remaining_svs[i]
            del pivot_sets[i]
            del sv_sizes[i]

        remaining_svs.extend(new_products)
        pivot_sets.extend([p.pivot_set() for p in new_products])
        sv_sizes.extend([p.size() for p in new_products])

        if len(remaining_svs) == 1:
            # --- Final vector produced, multiplication is finished -------
            return True, remaining_svs, pivot_sets, sv_sizes, union_sizes, intersection_sizes

        # ----  Update similarity matrices -----------
        # Decide whether to do a full recalculation or a cheaper update.
        # Recalculating is O(N*M + N^2) but is more accurate.
        # Updating is O(N*k) but can be slow if k (num_added) is large.
        # We recalculate if the number of new vectors is a significant
        # fraction of the total, as updating would be nearly as slow.
        num_added = len(pivot_sets) - (len(union_sizes) - len(sorted_indices))
        if num_added / len(remaining_svs) < 0.05:
            # ---- update if added few rows --------
            union_sizes, intersection_sizes = helpers.update_ps_unions_intersections(
                union_sizes, intersection_sizes, sorted_indices, pivot_sets
            )
        else:
            # ---- recalculate if added many rows --------
            union_sizes, intersection_sizes = helpers.calc_ps_unions_intersections(pivot_sets)

        return False, remaining_svs, pivot_sets, sv_sizes, union_sizes, intersection_sizes

    @staticmethod
    def multiply_all_vectors(
        state_vectors: List[StateVector], opt_config: dict, verbose: int = 0
    ) -> Tuple[StateVector, List[int]]:
        """
        Multiplies a list of StateVectors using an optimised clustering strategy.

        Parameters
        ----------
        state_vectors : List[StateVector]
            The list of StateVectors to multiply.
        opt_config : dict
            A dictionary of optimisation hyper-parameters.
        verbose : int, optional
            Verbosity level (0 for silent). Defaults to 0.

        Returns
        -------
        Tuple[StateVector, List[int]]
            - The final product StateVector.
            - A list of intermediate StateVector sizes for debugging.

        Notes
        -----
        This method uses a hybrid heuristic strategy:
        1. It first attempts to find a "predator-prey" relationship, where one
           vector is likely to significantly shrink several others.
        2. If no suitable predator is found, it falls back to a clustering
           strategy based on Jaccard similarity of pivot sets.
        """
        # --- Unpack optimisation parameters ---
        predator_base = opt_config.get("predator_base", 0.6)
        predator_threshold = opt_config.get("predator_threshold", 1.2)
        max_predator_size = opt_config.get("max_predator_size", 2)
        max_cluster_size = opt_config.get("max_cluster_size", 2)
        # --- End Unpack ---

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

        erase_line = False

        def _finalise():
            if erase_line:
                print(f"\r{' ' * 120}\r", end="")

        intermediate_sizes = []
        pivot_sets = [sv.pivot_set() for sv in remaining_svs]
        sv_sizes = [sv.size() for sv in remaining_svs]  # sizes of state vectors
        union_sizes, intersection_sizes = helpers.calc_ps_unions_intersections(pivot_sets)

        # ===== PREDATOR-PREY HEURISTIC ==============
        max_num_predator_prey_loops = (np.array(sv_sizes) <= max_predator_size).sum()
        counter = 0
        while len(remaining_svs) > 1:
            counter += 1
            if counter > max_num_predator_prey_loops:
                break  # Reached loop limit
            if len(remaining_svs) == 2:
                break  # Let main clustering loop handle the final simple multiplication

            sv0_idx, prey_indices = helpers.find_predator_prey(
                sv_sizes,
                intersection_sizes,
                base=predator_base,
                threshold=predator_threshold,
                max_predator_size=max_predator_size,
            )
            if sv0_idx is None:
                break  # No predator found, move to clustering

            if verbose > 1:
                p_size = sv_sizes[sv0_idx]
                n_prey = len(prey_indices)
                print(f"\r  - Predator found (size {p_size}), attacking {n_prey} prey... ", end=" " * 30)
                erase_line = True

            predator_sv = remaining_svs[sv0_idx]
            new_products = []
            indices_to_remove = set(prey_indices)
            indices_to_remove.add(sv0_idx)

            # --- Multiply all prey by predator ----
            deltas = []  # size reductions
            for i in prey_indices:
                product_sv = predator_sv * remaining_svs[i]
                delta = remaining_svs[i].size() - product_sv.size()
                deltas.append(delta)

                intermediate_sizes.append(product_sv.size())
                if product_sv.is_contradiction():
                    return StateVector(), intermediate_sizes
                new_products.append(product_sv)

            # ---- Update the state using the helper method ----
            indices_to_remove = list(indices_to_remove)
            is_finished, remaining_svs, pivot_sets, sv_sizes, union_sizes, intersection_sizes = (
                Engine._update_multiplication_state(
                    remaining_svs,
                    pivot_sets,
                    sv_sizes,
                    union_sizes,
                    intersection_sizes,
                    indices_to_remove,
                    new_products,
                )
            )

            if is_finished:
                _finalise()
                return remaining_svs[0], intermediate_sizes

            if not deltas or sum(deltas) <= 0:
                # ---- If the predator-prey step is no longer effective, exit.
                break
        # ====== END OF PREDATOR-PREY LOOP ===============

        # ======  JACCARD SIMILARITY CLUSTERING ==========
        while len(remaining_svs) > 1:
            if verbose > 1:
                max_sv_size = max([sv.size() for sv in remaining_svs])
                num_left = len(remaining_svs)
                print(f"\r  - Multiplying {num_left} vectors, max size: {max_sv_size}... ", end=" " * 30)

            if len(remaining_svs) == 2:
                product_sv = remaining_svs[0] * remaining_svs[1]
                intermediate_sizes.append(product_sv.size())
                _finalise()
                return product_sv, intermediate_sizes

            cluster_indices = helpers.find_next_cluster(pivot_sets, union_sizes, intersection_sizes, max_cluster_size)

            # ----- Multiply the vectors in the chosen cluster --------
            product_sv = remaining_svs[cluster_indices[0]]
            for i in cluster_indices[1:]:
                product_sv *= remaining_svs[i]
                intermediate_sizes.append(product_sv.size())
                if product_sv.is_contradiction():
                    _finalise()
                    return StateVector(), intermediate_sizes

            # ---- Update the state using the helper method ----
            is_finished, remaining_svs, pivot_sets, sv_sizes, union_sizes, intersection_sizes = (
                Engine._update_multiplication_state(
                    remaining_svs,
                    pivot_sets,
                    sv_sizes,
                    union_sizes,
                    intersection_sizes,
                    indices_to_remove=cluster_indices,
                    new_products=[product_sv],  # The single new product
                )
            )

            if is_finished:
                _finalise()
                return remaining_svs[0], intermediate_sizes
        _finalise()
        return remaining_svs[0], intermediate_sizes
