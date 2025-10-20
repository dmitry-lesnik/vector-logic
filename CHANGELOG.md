# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Future features will be listed here.

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
