"""
This module defines the TObject class.
"""

from typing import Iterable, Optional, List


class TObject:
    """
    Represents an immutable ternary object (1, 0, or X).

    The state is defined by two frozensets: one for indices that are fixed to 1 (True)
    and one for indices fixed to 0 (False). Indices not in either set are "Don't Care".
    """

    def __init__(
        self,
        ones: Optional[Iterable[int]] = None,
        zeros: Optional[Iterable[int]] = None,
        is_null: bool = False,
    ):
        """
        Initializes the TObject.

        Args:
            ones: An iterable of indices where the state is 1.
            zeros: An iterable of indices where the state is 0.
            is_null: A flag to create a null object, which represents a contradiction.

        Raises:
            ValueError: If 'is_null' is True and 'ones' or 'zeros' are also provided.
        """
        if is_null and (ones is not None or zeros is not None):
            raise ValueError("Cannot specify 'ones' or 'zeros' when 'is_null' is True.")

        self._ones: frozenset[int] = frozenset(ones) if ones is not None else frozenset()
        self._zeros: frozenset[int] = frozenset(zeros) if zeros is not None else frozenset()

        if not self._ones.isdisjoint(self._zeros):
            self._ones = frozenset()
            self._zeros = frozenset()
            self._is_null: bool = True
        else:
            self._is_null = is_null

        self._pivot_set = None

    @property
    def ones(self) -> frozenset[int]:
        return self._ones

    @property
    def zeros(self) -> frozenset[int]:
        return self._zeros

    @property
    def is_null(self) -> bool:
        return self._is_null

    @property
    def is_trivial(self) -> bool:
        return not self.is_null and not self._ones and not self._zeros

    @property
    def pivot_set(self) -> frozenset[int]:
        if self._pivot_set is None:
            if self._is_null:
                self._pivot_set = frozenset()
            else:
                self._pivot_set = self._ones.union(self._zeros)
        return self._pivot_set

    def var_value(self, index: int) -> int:
        """
        Checks the value of a single variable index.

        Args:
            index: The variable index to check.

        Returns:
            1 if the index is in the 'ones' set.
            0 if the index is in the 'zeros' set.
            -1 if the index is a "Don't Care".
        """
        if index in self._ones:
            return 1
        if index in self._zeros:
            return 0
        return -1

    def to_string(self, max_index: Optional[int] = None) -> str:
        """
        Displays the TObject as a string of 0s, 1s, and dashes.

        Args:
            max_index: The largest index to display. If None, it defaults to the
                largest index in the object.

        Returns:
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
            elif i < 1:
                result.append("*")
        for i in self._zeros:
            if 1 <= i <= effective_max:
                result[i - 1] = "0"
            elif i < 1:
                result.append("*")

        return " ".join(result)

    def __mul__(self, other: "TObject") -> "TObject":
        """
        Calculates the product of two TObjects.
        """
        if not isinstance(other, TObject):
            return NotImplemented

        if self.is_null or other.is_null:
            return TObject(is_null=True)

        new_ones = self.ones.union(other.ones)
        new_zeros = self.zeros.union(other.zeros)

        return TObject(ones=new_ones, zeros=new_zeros)

    def negate_variables(self, variable_indices: List[int] | int) -> "TObject":
        """
        Returns a new TObject with specified variables negated.
        """
        if self._is_null:
            return TObject(is_null=True)

        indices = {variable_indices} if isinstance(variable_indices, int) else set(variable_indices)

        new_ones = (self.ones.difference(indices)).union(self.zeros.intersection(indices))
        new_zeros = (self.zeros.difference(indices)).union(self.ones.intersection(indices))

        return TObject(ones=new_ones, zeros=new_zeros)

    def remove_variables(self, variable_indices: List[int] | int) -> "TObject":
        """
        Returns a new TObject with specified variables removed.
        """
        if self._is_null:
            return TObject(is_null=True)
        indices = {variable_indices} if isinstance(variable_indices, int) else set(variable_indices)

        new_ones = self.ones.difference(indices)
        new_zeros = self.zeros.difference(indices)

        return TObject(ones=new_ones, zeros=new_zeros)

    def reduce(self, other: "TObject") -> Optional["TObject"]:
        """
        Reduces two TObjects if they are adjacent.

        Two TObjects are adjacent if they differ by exactly one index, where one has a 1
        and the other has a 0, and are otherwise identical.

        Returns:
            A new, reduced TObject if reducible, otherwise None.
        """
        if not isinstance(other, TObject):
            return None

        # ======= check length before doing set operations =========
        ones_diff_len = len(self._ones) - len(other.ones)
        if abs(ones_diff_len) == 1:
            zeros_diff_len = len(self._zeros) - len(other.zeros)
            if ones_diff_len == 1:
                if zeros_diff_len != -1:
                    return None
            elif ones_diff_len == -1:
                if zeros_diff_len != 1:
                    return None
        else:
            return None

        # ====== check set differences =================
        ones_diff = self.ones.symmetric_difference(other.ones)

        if len(ones_diff) == 1:
            zeros_diff = self.zeros.symmetric_difference(other.zeros)
            if ones_diff == zeros_diff:
                (idx,) = ones_diff
                if idx in self.ones:
                    new_ones = other.ones
                    new_zeros = self.zeros
                else:
                    new_ones = self.ones
                    new_zeros = other.zeros
                return TObject(ones=new_ones, zeros=new_zeros)

        return None

    def is_superset(self, other: "TObject") -> int:
        """
        Checks for a superset relationship between two TObjects.

        A TObject is a superset of another if it is more general (has fewer constraints) or identical.

        - Returns 1 if this TObject is a superset of the other, or if this equals the other.
        - Returns -1 if the other TObject is a superset of this one.
        - Returns 0 otherwise.
        """
        is_self_superset = self.ones.issubset(other.ones) and self.zeros.issubset(other.zeros)
        if is_self_superset:
            return 1

        is_other_superset = other.ones.issubset(self.ones) and other.zeros.issubset(self.zeros)
        if is_other_superset:
            return -1
        return 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TObject):
            return NotImplemented

        if self.is_null:
            return other.is_null

        return not other.is_null and self.ones == other.ones and self.zeros == other.zeros

    def __hash__(self) -> int:
        return hash((self.is_null, self.ones, self.zeros))

    def __repr__(self):
        if self.is_null:
            return "TObject(is_null=True)"
        return f"TObject(ones={set(self.ones)}, zeros={set(self.zeros)})"

    def __lt__(self, other: object) -> bool:
        """Compares two TObjects for sorting purposes."""
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
