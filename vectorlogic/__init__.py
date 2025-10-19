"""
Vector Logic Package.

This package provides a simple rule engine based on state vector representation.
It allows for defining logical rules and performing logical inference.

The main components exposed are:
- Engine: The main interface for creating and managing a rule engine instance.
- StateVector: A core data structure representing a set of logical states.
- TObject: A fundamental building block representing a single row of a state vector.
"""

__version__ = "0.1.1"
__author__ = "Dmitry Lesnik"
__credits__ = "Stratyfy Inc"

from .engine import Engine
from .state_vector import StateVector
from .t_object import TObject
