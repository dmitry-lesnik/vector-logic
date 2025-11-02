# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Future features will be listed here.

## [0.4.0] - 2025-11-??

### Added

- **Engine Verbosity**: Added an optional `verbose` parameter to the `Engine` constructor. When enabled, it provides real-time progress updates during the `compile()` process, showing the number of remaining state vectors and the maximum size of any vector. This is useful for debugging performance on large or complex rule sets.
- **StateVector Iterator**: Implemented `__iter__` and `__getitem__` in the `StateVector` class, allowing it to be used as a standard Python iterable. This enables direct iteration over its internal `TObject` instances (e.g., `for t_obj in state_vector:`).
- **InferenceResult Iterator**: Added an `iter_dicts()` method to the `InferenceResult` class, providing a convenient way to iterate over the resulting states as dictionaries.

### Changed

- **Iterator Caching**: The `Engine.valid_set_iter_dicts()` method now caches the inverted `index_to_name` mapping, improving performance for repeated iterations over the valid set.

## [0.3.0] - 2025-11-01

### Added

- **Predator-Prey Optimisation Heuristic**: Implemented a new optimisation phase in `Engine.multiply_all_vectors` that
  runs before Jaccard similarity clustering. This "predator-prey" logic uses small `StateVector` instances (e.g., from
  evidence) to significantly reduce the size of larger "prey" vectors, dramatically improving compilation performance on
  many rule sets.

- **Optimisation Parameter API**: Added an `.opt_config` property to the `Engine` class. This allows advanced users to
  tune the optimisation heuristics (e.g. `max_predator_size`, `max_cluster_size`) by modifying internal attributes like
  `_opt_max_predator_size` on an engine instance.

### Changed

- Refactored the `Engine.multiply_all_vectors` method, splitting the core optimization logic into smaller, internal
  static methods (`_update_multiplication_state`) to improve readability and maintainability.

## [0.2.0] - 2025-10-30

### Added

- API: export t-object as a dictionary
- API: iterate through rows of a state vector represented as dictionaries

## [0.1.2] - 2025-10-20

### Changed

- Updated pyproject.toml to conform to the PEP 621 standard, resolving packaging warnings.

- Minor improvements to `README.md`.

## [0.1.1] - 2025-10-19

### Changed

- Fixed typos in README and CONTRIBUTING

## [0.1.0] - 2025-10-19

### Added

- **Initial Release**
- Core data structures: `TObject` for ternary logic states and `StateVector` for collections of states.
- Main `Engine` class to manage variables, rules, compilation, and inference.
- `RuleParser` using `pyparsing` to convert logical rule strings into an Abstract Syntax Tree (AST).
- `RuleConverter` to transform complex ASTs into simplified `StateVector` representations.
- Support for a full suite of logical operators: `&&` (AND), `||` (OR), `^^` (XOR), `=>` (IMPLIES), `<=` (IS IMPLIED
  BY), and `=` (EQUIVALENCE).
- `StateVector` multiplication (`*`) for logical intersection and `simplify()` for reduction.
- Optimized `_adjacency_reduction` method using grouping to avoid O(N²) complexity.
- Heuristic-based `compile()` method that uses Jaccard similarity to optimize the multiplication order of rules,
  improving performance.
- `predict()` method for performing inference with new evidence without modifying the compiled knowledge base.
- Comprehensive test suite using `pytest`.
- Example usage scripts demonstrating basic functionality and logical proofs.
- Project documentation: `README.md`, `CONTRIBUTING.md`, and `CODE_OF_CONDUCT.md`.
