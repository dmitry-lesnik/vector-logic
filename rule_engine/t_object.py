"""
Defines the core TObject class for ternary logic representation.

This module provides the `TObject` class, which is a fundamental data structure
in the rule engine. It represents an immutable ternary object, where each
variable can be in one of three states: True (1), False (0), or Schrödinger's cat (-).
"""

from typing import Iterable, Optional, List, Tuple


class TObject:
    """
    Represents an immutable ternary object, equivalent to a conjunction of literals.

    A `TObject` can be conceptualized as a subset of a truth table, filtered by
    a set of constraints. These constraints fix specific variables to be either
    1 (True) or 0 (False), while unconstrained variables can take any value.
    This makes it logically equivalent to a conjunction of literals. For example,
    `TObject(ones={1}, zeros={3})` corresponds to `v1 AND (NOT v3)`.

    The internal state is defined by two frozensets: `ones` for indices fixed
    to 1 and `zeros` for indices fixed to 0. An object with conflicting
    constraints (i.e., overlapping `ones` and `zeros` sets) represents a
    logical contradiction and is automatically converted to a "null" state.

    Conversely, a `TObject` initialized with no constraints (empty `ones` and
    `zeros` sets) is a "trivial" object. This represents a tautology, as it
    allows any variable to be assigned any value, effectively matching all
    possible states.

    Parameters
    ----------
    ones : Iterable[int], optional
        An iterable of 1-based indices where the state is 1 (True).
        Defaults to None.
    zeros : Iterable[int], optional
        An iterable of 1-based indices where the state is 0 (False).
        Defaults to None.
    is_null : bool, optional
        If True, creates a null object representing a contradiction.
        Defaults to False.

    Attributes
    ----------
    ones : frozenset[int]
        The set of indices fixed to 1.
    zeros : frozenset[int]
        The set of indices fixed to 0.
    is_null : bool
        True if the object represents a contradiction.

    Raises
    ------
    ValueError
        If `is_null` is True and `ones` or `zeros` are also provided.
    """

    def __init__(
        self,
        ones: Optional[Iterable[int]] = None,
        zeros: Optional[Iterable[int]] = None,
        is_null: bool = False,
    ):
        if is_null and (ones is not None or zeros is not None):
            raise ValueError("Cannot specify 'ones' or 'zeros' when 'is_null' is True.")

        self._ones: frozenset[int] = frozenset(ones) if ones is not None else frozenset()
        self._zeros: frozenset[int] = frozenset(zeros) if zeros is not None else frozenset()

        if not self._ones.isdisjoint(self._zeros):
            # A contradiction was created (e.g., index 1 is both 0 and 1).
            # This becomes a null object.
            self._ones = frozenset()
            self._zeros = frozenset()
            self._is_null: bool = True
        else:
            self._is_null = is_null

        self._pivot_set: Optional[frozenset[int]] = None

    @property
    def ones(self) -> frozenset[int]:
        """The frozenset of indices fixed to 1 (True)."""
        return self._ones

    @property
    def zeros(self) -> frozenset[int]:
        """The frozenset of indices fixed to 0 (False)."""
        return self._zeros

    @property
    def is_null(self) -> bool:
        """Return True if the TObject represents a contradiction."""
        return self._is_null

    @property
    def is_trivial(self) -> bool:
        """
        Return True if the TObject represents a tautology.

        A trivial TObject has no constraints (empty `zeros` and `ones`)
        and is not null.
        """
        return not self.is_null and not self._ones and not self._zeros

    @property
    def pivot_set(self) -> frozenset[int]:
        """
        Return the set of all constrained variable indices (both 1s and 0s).
        """
        if self._pivot_set is None:
            if self._is_null:
                self._pivot_set = frozenset()
            else:
                self._pivot_set = self._ones.union(self._zeros)
        return self._pivot_set

    def var_value(self, index: int) -> int:
        """
        Check the value of a single variable index.

        Parameters
        ----------
        index : int
            The 1-based variable index to check.

        Returns
        -------
        int
            - 1 if the index is in the 'ones' set.
            - 0 if the index is in the 'zeros' set.
            - -1 if the index is unconstrained.
            - Raises ValueError if called on a null TObject.

        Raises
        ------
        ValueError
            If called on a null TObject.
        """
        if self._is_null:
            raise ValueError("Cannot determine variable value for a null TObject.")
        if index in self._ones:
            return 1
        if index in self._zeros:
            return 0
        return -1

    def to_string(self, max_index: Optional[int] = None) -> str:
        """
        Generate a string representation of the TObject.

        The string consists of 0s, 1s, and dashes for unconstrained indices.

        Parameters
        ----------
        max_index : int, optional
            The largest index to display for alignment. If None, it defaults
            to the largest constrained index in the object.

        Returns
        -------
        str
            The string representation (e.g., "1 - 0 -").
        """
        if self.is_null:
            return "null"
        if self.is_trivial:
            return "---"

        effective_max = max_index if max_index is not None else max(self.pivot_set, default=0)

        result = ["-"] * effective_max
        for i in self._ones:
            if 1 <= i <= effective_max:
                result[i - 1] = "1"
        for i in self._zeros:
            if 1 <= i <= effective_max:
                result[i - 1] = "0"

        return " ".join(result)

    def __mul__(self, other: "TObject") -> "TObject":
        """
        Calculate the product (intersection) of two TObjects.

        The product combines the constraints of both objects. If the constraints
        are contradictory (e.g., one requires index `i` to be 1 and the other
        requires it to be 0), the result is a null `TObject`.

        Parameters
        ----------
        other : TObject
            The `TObject` to multiply with this one.

        Returns
        -------
        TObject
            A new `TObject` representing the product.
        """
        if not isinstance(other, TObject):
            return NotImplemented

        if self.is_null or other.is_null:
            return TObject(is_null=True)

        new_ones = self.ones.union(other.ones)
        new_zeros = self.zeros.union(other.zeros)

        # The constructor handles the check for contradictions (ones/zeros overlap)
        return TObject(ones=new_ones, zeros=new_zeros)

    def negate_variables(self, variable_indices: Tuple[List[int], int]) -> "TObject":
        """
        Return a new TObject with specified variables negated.

        For each specified index, a 1 becomes a 0, and a 0 becomes a 1.
        Unconstrained variables remain unchanged.

        Parameters
        ----------
        variable_indices : Tuple[List[int], int]
            A single index or a list of indices to negate.

        Returns
        -------
        TObject
            A new `TObject` with the specified variables negated.
        """
        if self._is_null:
            return TObject(is_null=True)

        indices = {variable_indices} if isinstance(variable_indices, int) else set(variable_indices)

        new_ones = (self.ones.difference(indices)).union(self.zeros.intersection(indices))
        new_zeros = (self.zeros.difference(indices)).union(self.ones.intersection(indices))

        return TObject(ones=new_ones, zeros=new_zeros)

    def remove_variables(self, variable_indices: Tuple[List[int], int]) -> "TObject":
        """
        Return a new TObject with specified variables removed (set unconstrained).

        This operation makes the `TObject` more general by removing constraints
        on the specified variables.

        Parameters
        ----------
        variable_indices : Tuple[List[int], int]
            A single index or a list of indices to remove.

        Returns
        -------
        TObject
            A new `TObject` with the constraints on the specified variables removed.
        """
        if self._is_null:
            return TObject(is_null=True)
        indices = {variable_indices} if isinstance(variable_indices, int) else set(variable_indices)

        new_ones = self.ones.difference(indices)
        new_zeros = self.zeros.difference(indices)

        return TObject(ones=new_ones, zeros=new_zeros)

    def reduce(self, other: "TObject") -> Optional["TObject"]:
        """
        Reduce two TObjects if they are adjacent.

        Two TObjects are "adjacent" if they differ by exactly one index, where
        one has a 1 and the other has a 0, and are otherwise identical. The
        reduction removes this differing constraint, creating a more general
        TObject.

        Example: `(1, 0, -)` and `(1, 1, -)` can be reduced to `(1, -, -)`.

        Parameters
        ----------
        other : TObject
            The `TObject` to attempt reduction with.

        Returns
        -------
        Optional[TObject]
            A new, reduced `TObject` if reducible, otherwise `None`.
        """
        if not isinstance(other, TObject):
            return None

        ones_diff = self.ones.symmetric_difference(other.ones)
        if len(ones_diff) == 1:
            zeros_diff = self.zeros.symmetric_difference(other.zeros)
            if ones_diff == zeros_diff:
                # This confirms they differ at exactly one position, one being
                # a 1 and the other a 0.
                (idx,) = ones_diff
                if idx in self.ones:
                    # self has the 1, other has the 0
                    new_ones = other.ones
                    new_zeros = self.zeros
                else:
                    # other has the 1, self has the 0
                    new_ones = self.ones
                    new_zeros = other.zeros
                return TObject(ones=new_ones, zeros=new_zeros)

        return None

    def is_superset(self, other: "TObject") -> int:
        """
        Check for a superset relationship between two TObjects.

        A TObject is a "superset" of another if it is more general or identical.
        This means all constraints in the superset must also be present in the
        subset. For example, `(1, -, -)` is a superset of `(1, 0, -)`.

        Parameters
        ----------
        other : TObject
            The `TObject` to compare against.

        Returns
        -------
        int
            -  1 if this TObject is a superset of `other` (or they are equal).
            - -1 if `other` is a superset of this TObject.
            -  0 otherwise (no superset relationship).
        """
        is_self_superset = self.ones.issubset(other.ones) and self.zeros.issubset(other.zeros)
        if is_self_superset:
            # This case also handles equality, where both are supersets of each other.
            return 1

        is_other_superset = other.ones.issubset(self.ones) and other.zeros.issubset(self.zeros)
        if is_other_superset:
            return -1
        return 0

    def __eq__(self, other: object) -> bool:
        """Check for equality between two TObjects."""
        if not isinstance(other, TObject):
            return NotImplemented

        if self.is_null:
            return other.is_null

        return not other.is_null and self.ones == other.ones and self.zeros == other.zeros

    def __hash__(self) -> int:
        """Return a hash based on the immutable state."""
        return hash((self.is_null, self.ones, self.zeros))

    def __repr__(self) -> str:
        """Return the canonical string representation."""
        if self.is_null:
            return "TObject(is_null=True)"
        return f"TObject(ones={set(self.ones)}, zeros={set(self.zeros)})"

    def __lt__(self, other: object) -> bool:
        """
        Compare two TObjects for sorting purposes.

        This provides a stable, canonical ordering for TObjects, which is
        useful when they are stored in sorted collections. The comparison is
        based on the sorted lists of `ones` and `zeros` indices. Null objects
        are considered "less than" non-null objects.
        """
        if not isinstance(other, TObject):
            return NotImplemented

        if self.is_null:
            return not other.is_null
        if other.is_null:
            return False

        # Sort by 'ones' then 'zeros' for a stable, canonical order.
        self_tuple = (sorted(list(self.ones)), sorted(list(self.zeros)))
        other_tuple = (sorted(list(other.ones)), sorted(list(other.zeros)))

        return self_tuple < other_tuple
