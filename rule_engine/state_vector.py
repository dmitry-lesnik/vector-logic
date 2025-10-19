"""
Defines the StateVector class, a core data structure for the rule engine.

This module provides the `StateVector` class, which represents a collection of
`TObject` instances. It is the primary data structure for representing logical
rules and knowledge bases. The class provides methods for logical operations,
simplification, and analysis.
"""

from collections import defaultdict
from typing import List, Optional, Set, Tuple

from .t_object import TObject


class StateVector:
    """
    Represents an immutable collection of TObjects, defining a set of valid logical states.

    This class is the primary data structure for representing logical rules and
    knowledge bases. A StateVector can be thought of as a Disjunctive Normal
    Form (DNF), where each `TObject` is a conjunction of literals, and the
    vector itself is a disjunction of these conjunctions. Operations like
    multiplication and simplification are performed by creating new `StateVector`
    instances, ensuring immutability.

    Attributes
    ----------
    _t_objects : tuple[TObject, ...]
        An immutable tuple of the TObjects comprising the vector.
    _pivot_set_cache : Optional[Set[int]]
        A cached set of all active variable indices in the vector.
    """

    def __init__(self, t_objects: Optional[List[TObject]] = None):
        """
        Initializes the StateVector.

        Parameters
        ----------
        t_objects : List[TObject], optional
            A list of TObjects to initialize the state vector with. The
            interpretation of the input is as follows:
            - `StateVector()` or `StateVector([])`: Creates an empty vector,
              which represents an infeasible state (a logical contradiction).
            - `StateVector([TObject()])`: Creates a trivial vector containing
              a single, unconstrained TObject. This represents a tautology,
              where all possible states are allowed.
            - `StateVector([...])`: A standard vector with one or more
              constrained TObjects.
        """
        self._t_objects: tuple[TObject, ...] = tuple(t_objects) if t_objects is not None else tuple()
        self._pivot_set_cache: Optional[Set[int]] = None

    def __eq__(self, other: "StateVector") -> bool:
        """
        Check if two StateVectors are structurally identical.

        Equality is determined by comparing the sorted tuples of TObjects.
        This is primarily useful for debugging and testing.

        .. warning::
           This method does not check for logical equivalence. Two logically
           equivalent but structurally different StateVectors will not be
           considered equal.

        Parameters
        ----------
        other : StateVector
            The StateVector to compare with.

        Returns
        -------
        bool
            True if the StateVectors contain the exact same TObjects, False otherwise.
        """
        if not isinstance(other, StateVector):
            return NotImplemented
        # Sorting is necessary because the internal order of TObjects is not guaranteed.
        return sorted(self._t_objects) == sorted(other._t_objects)

    def __mul__(self, other: "StateVector") -> "StateVector":
        """
        Calculate the product of two StateVectors.

        The product corresponds to the logical AND operation, finding the
        intersection of valid states between the two vectors. The resulting
        vector is pragmatically simplified to manage performance during
        chained multiplications.

        Parameters
        ----------
        other : StateVector
            The StateVector to multiply with.

        Returns
        -------
        StateVector
            A new, simplified StateVector representing the product.
        """
        if not isinstance(other, StateVector):
            return NotImplemented

        if self.is_contradiction() or other.is_contradiction():
            return StateVector()  # Product with a contradiction is a contradiction

        if self.is_trivial():
            return other
        if other.is_trivial():
            return self

        new_t_objects = [prod for t1 in self._t_objects for t2 in other._t_objects if not (prod := t1 * t2).is_null]

        # A single iteration of simplification is a pragmatic choice for performance
        # during long compilation chains.
        return StateVector(new_t_objects).simplify(max_num_iter=1)

    def __repr__(self) -> str:
        """Return an unambiguous string representation of the StateVector."""
        return f"StateVector(t_objects={self._t_objects!r})"

    def size(self) -> int:
        """
        Return the number of TObjects in the state vector.

        Returns
        -------
        int
            The count of TObjects.
        """
        return len(self._t_objects)

    def pivot_set(self) -> Set[int]:
        """
        Calculate the union of pivot sets of all TObjects in the vector.

        The pivot set contains all variable indices that are explicitly
        constrained (set to 0 or 1) in at least one TObject. The result is
        cached after the first calculation for efficiency.

        Returns
        -------
        Set[int]
            A set of all active variable indices in the StateVector.
        """
        if self._pivot_set_cache is None:
            if not self._t_objects:
                self._pivot_set_cache = set()
            else:
                all_indices = set()
                for t_obj in self._t_objects:
                    all_indices.update(t_obj.pivot_set)
                self._pivot_set_cache = all_indices
        return self._pivot_set_cache

    def _subsumption_reduction(self) -> Tuple["StateVector", bool]:
        """
        Perform one pass of subsumption reduction.

        This method removes more specific TObjects that are covered by more
        general ones. For example, the TObject representing `(1, -, -)`
        subsumes (is a superset of) `(1, 1, -)`, so the latter can be removed
        without changing the logical meaning of the StateVector.

        Returns
        -------
        Tuple[StateVector, bool]
            A tuple containing:
            - The new, reduced StateVector.
            - A boolean flag that is True if any reduction occurred.
        """
        num_t_objects = len(self._t_objects)
        if num_t_objects < 2:
            return self, False

        removed = [False] * num_t_objects
        was_modified = False
        for i in range(num_t_objects):
            if removed[i]:
                continue
            for j in range(i + 1, num_t_objects):
                if removed[j]:
                    continue

                superset_status = self._t_objects[i].is_superset(self._t_objects[j])
                if superset_status == 1:  # t_obj i is superset of t_obj j
                    removed[j] = True
                    was_modified = True
                elif superset_status == -1:  # t_obj j is superset of t_obj i
                    removed[i] = True
                    was_modified = True
                    break  # t_obj i is removed, move to the next i
        if was_modified:
            final_t_objects = [t for i, t in enumerate(self._t_objects) if not removed[i]]
            return StateVector(final_t_objects), True
        return self, False

    def _adjacency_reduction(self, max_num_iter: Optional[int] = 1) -> Tuple["StateVector", bool]:
        """
        Perform an optimized adjacency reduction.

        This method combines adjacent TObjects using the rule `(A & B) | (A & !B) = A`.
        For example, `(1, 0, -)` and `(1, 1, -)` can be reduced to `(1, -, -)`.

        Parameters
        ----------
        max_num_iter : int, optional
            The maximum number of reduction passes to perform. Defaults to 1.

        Returns
        -------
        Tuple[StateVector, bool]
            A tuple containing:
            - The new, reduced StateVector.
            - A boolean flag that is True if any reduction occurred.
        """
        t_objects = list(self._t_objects)
        was_modified = False

        num_iter = 0
        while max_num_iter is None or num_iter < max_num_iter:
            num_iter += 1

            was_reduced_this_iter = False
            num_t_objects = len(t_objects)
            removed = [False] * num_t_objects
            new_t_objects = []

            # Group TObjects by structural properties to avoid O(N^2) comparisons.
            # The key is a tuple of (pivot_set, ones_length). T-objects are only
            # reducible if their pivot sets coincide and one has one more 'one'.
            groups = defaultdict(list)
            for i, t_obj in enumerate(t_objects):
                key = (t_obj.pivot_set, len(t_obj.ones))
                groups[key].append(i)

            for key, group1_indices in groups.items():
                pivot_set, ones_len = key

                # Find the compatible group: same pivot set, one fewer 'one'.
                compatible_key = (pivot_set, ones_len - 1)
                group2_indices = groups.get(compatible_key)

                if not group2_indices:
                    continue

                # Iterate through pairs from these two compatible groups.
                for i in group1_indices:
                    if removed[i]:
                        continue
                    for j in group2_indices:
                        if removed[j]:
                            continue

                        reduced_obj = t_objects[i].reduce(t_objects[j])

                        if reduced_obj is not None:
                            removed[i] = True
                            removed[j] = True
                            new_t_objects.append(reduced_obj)
                            was_reduced_this_iter = True
                            break  # t_obj i has been reduced, move to next in group1

            if was_reduced_this_iter:
                t_objects = [t_obj for i, t_obj in enumerate(t_objects) if not removed[i]] + new_t_objects
                was_modified = True
            else:
                break  # No more reductions possible in a full pass

        if was_modified:
            return StateVector(t_objects=t_objects), True
        else:
            return self, False

    def simplify(self, max_num_iter: Optional[int] = 1, reduce_subsumption: bool = False) -> "StateVector":
        """
        Perform a full simplification of the StateVector.

        This method repeatedly applies adjacency reduction and optionally subsumption
        reduction to minimize the number of TObjects in the vector.

        .. note:: The reduction is not canonical, and hence doesn't guarantee
           a unique representation for logically equivalent StateVectors.
           (see https://arxiv.org/abs/2509.10326 for details)

        Parameters
        ----------
        max_num_iter : int, optional
            The maximum number of passes for the adjacency reduction loop.
            If None, it runs until a fixed point is reached. Defaults to 1.
        reduce_subsumption : bool, optional
            If True, subsumption reduction is also performed, which is more
            computationally expensive but can yield a smaller vector.
            Defaults to False.

        Returns
        -------
        StateVector
            A new, simplified StateVector.
        """
        # Remove nulls and duplicates. `dict.fromkeys` preserves order.
        t_objects = list(dict.fromkeys(t for t in self._t_objects if not t.is_null))

        # If any TObject is trivial (fully unconstrained), it covers all possible states.
        if any(t_obj.is_trivial for t_obj in t_objects):
            return StateVector([TObject()])

        current_sv = StateVector(t_objects)

        if reduce_subsumption:
            current_sv, _ = current_sv._subsumption_reduction()

        was_modified = True
        while was_modified:
            current_sv, was_modified = current_sv._adjacency_reduction(max_num_iter=max_num_iter)

        if reduce_subsumption:
            current_sv, _ = current_sv._subsumption_reduction()

        return current_sv

    def negate_variables(self, variable_indices: Tuple[List[int], int]) -> "StateVector":
        """
        Return a new StateVector with specified variables negated.

        Parameters
        ----------
        variable_indices : Tuple[List[int], int]
            A single index or a list of indices to negate.

        Returns
        -------
        StateVector
            A new StateVector with the variables negated in each TObject.
        """
        new_t_objects = [t.negate_variables(variable_indices) for t in self._t_objects]
        return StateVector(new_t_objects)

    def remove_variables(self, variable_indices: Tuple[List[int], int]) -> "StateVector":
        """
        Return a new StateVector with specified variables removed.

        This operation makes the StateVector more general by turning the specified
        variable indices into unconstrained states in all TObjects.

        Parameters
        ----------
        variable_indices : Tuple[List[int], int]
            A single index or a list of indices to remove.

        Returns
        -------
        StateVector
            A new StateVector with the variables removed.
        """
        new_t_objects = [t.remove_variables(variable_indices) for t in self._t_objects]
        return StateVector(new_t_objects)

    def is_contradiction(self) -> bool:
        """
        Check if the state vector represents a contradiction.

        A contradiction occurs when the set of valid states is empty, meaning
        the vector contains no TObjects.

        Returns
        -------
        bool
            True if the vector contains no TObjects, False otherwise.
        """
        return not self._t_objects

    def is_trivial(self) -> bool:
        """
        Check if the state vector is trivial (a tautology).

        A state vector is trivial if it contains exactly one TObject, and that
        TObject is itself trivial (represents all possible states, i.e., no
        constraints).

        Returns
        -------
        bool
            True if the state vector is trivial, False otherwise.
        """
        return len(self._t_objects) == 1 and self._t_objects[0].is_trivial

    def var_value(self, index: int) -> int:
        """
        Check the consolidated value of a variable across all TObjects.

        This method determines if a variable has a consistent state (always 1 or
        always 0) across all possible states represented by the vector.

        Parameters
        ----------
        index : int
            The 1-based index of the variable to check.

        Returns
        -------
        int
            - 1 if the variable is consistently True across all TObjects.
            - 0 if the variable is consistently False across all TObjects.
            - -1 if the variable is mixed (sometimes True, sometimes False) or
              is unconstrained in any TObject.

        Raises
        ------
        ValueError
            If called on a contradictory (empty) StateVector.
        """
        if not self._t_objects:
            raise ValueError("Cannot determine variable value for an empty StateVector.")

        first_value = self._t_objects[0].var_value(index)
        if first_value == -1:
            # If it's unconstrained in any TObject, the consolidated value is undetermined.
            return -1

        for t_obj in self._t_objects[1:]:
            if t_obj.var_value(index) != first_value:
                # If the value differs in any other TObject, it's mixed.
                return -1
        return first_value

    def to_string(self, max_index: Optional[int] = None, indent: int = 0, print_brackets: bool = True) -> str:
        """
        Generate a formatted string representation of the StateVector.

        Parameters
        ----------
        max_index : int, optional
            The largest index to display for alignment. If None, it is
            auto-calculated. Defaults to None.
        indent : int, optional
            The number of spaces to indent the output. Defaults to 0.
        print_brackets : bool, optional
            If True, encloses the output in curly brackets. Defaults to True.

        Returns
        -------
        str
            The formatted string.
        """
        base_indent_str = " " * indent

        if not self._t_objects:
            if print_brackets:
                return f"{base_indent_str}{{ Contradiction }}"
            return f"{base_indent_str}Contradiction"

        effective_max_index = max_index
        if effective_max_index is None:
            all_indices = self.pivot_set()
            effective_max_index = max(all_indices) if all_indices else 0

        content_indent_str = " " * (indent + 4) if print_brackets else base_indent_str

        string_lines = [
            content_indent_str + t_obj.to_string(max_index=effective_max_index) for t_obj in self._t_objects
        ]
        content = "\n".join(string_lines)

        if print_brackets:
            return f"{base_indent_str}{{\n{content}\n{base_indent_str}}}"
        return content

    def print(self, max_index: Optional[int] = None, indent: int = 0, print_brackets: bool = True):
        """
        Print the formatted string representation of the StateVector.

        Parameters
        ----------
        max_index : int, optional
            The largest index to display for alignment. Defaults to None.
        indent : int, optional
            The number of spaces to indent the output. Defaults to 0.
        print_brackets : bool, optional
            If True, encloses the output in curly brackets. Defaults to True.
        """
        print(self.to_string(max_index=max_index, indent=indent, print_brackets=print_brackets))
