# Rule Engine Library

A lightweight and extensible Python library for defining and evaluating propositional logic rules. It leverages State
Algebra concepts to represent and manipulate logical expressions, allowing for powerful rule compilation and state
analysis.

This library is for developers who need a simple, transparent, and lightweight rule engine. If you find that more
sophisticated tools like Binary Decision Diagrams (BDDs) or SAT solvers are overkill for your use case, this engine
provides an ideal middle ground.

The implementation is based on the theory of **State Algebra**, which reframes propositional logic in simple algebraic
terms. The core concepts are easy to understand, and the codebase is small and self-contained, making it a great
starting point for any simple rule-based application.

## Features

* **Expressive Rule Syntax**: Define rules using a natural, human-readable syntax (e.g., `x1 = (x2 && x3)`).
* **Rule Compilation**: Combine multiple rules and evidence into a single, consolidated "Valid Set" representing all
  possible valid states.
* **Inference and Prediction**: Test new evidence against the compiled knowledge base to make predictions and check for
  contradictions.
* **State Analysis**: Query the state of any variable to determine if it is always true, always false, or can vary.
* **Lightweight & Transparent**: A small, focused codebase with no heavy dependencies, making it easy to understand and
  extend.

> **A Note on Performance:** The efficiency of this approach relies on heuristic optimizations for the order of rule
> multiplication. This implementation includes a basic but effective heuristic that performs well for systems with up to
> ~100 variables and a similar number of rules. For more challenging tasks, performance can be improved by implementing
> more advanced clustering heuristics.

## Installation

This project uses Poetry for dependency management.

1. Install Poetry:
   Follow the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

2. Clone the repository:

    ```bash
    git clone https://github.com/dmitry-lesnik/rule-engine.git
    cd rule-engine
    ```

3. Install dependencies:
    ```bash
    poetry install
    ```

## Usage

The core of the library is the `Engine` class, which allows you to define variables, add logical rules, and then compile
them into a consolidated `StateVector`.

```python
from rule_engine import Engine

# --- Understanding Compilation ---
#
# The rule-engine offers two main approaches for inference, and choosing
# the right one depends on your use case.
#
# 1. Pre-compiling the Knowledge Base (`engine.compile()`):
#
#    - What it does: Multiplies all added rules together to create a single,
#      optimized StateVector called the "valid set".
#    - When to use it: When you need to run multiple predictions against the
#      same set of rules.
#    - Trade-off: The initial compilation can be slow if the valid set is
#      very large, but subsequent predictions (`engine.predict()`) will be
#      extremely fast because they only need to multiply with this single,
#      pre-computed valid set.
#
# 2. On-the-Fly Prediction (using `engine.predict()` on an un-compiled engine):
#
#    - What it does: Multiplies all rules from scratch every time you call
#      `.predict()`, including the evidence for that specific prediction.
#    - When to use it: When you only need to run one or a few predictions.
#    - Trade-off: This can be faster for a single query because a restrictive
#      piece of evidence can cause the intermediate StateVectors to become
#      very small, avoiding the creation of a potentially huge valid set.
#

# 1. Define the variables involved in your rules
variables = ["x1", "x2", "x3", "x4"]

# 2. Create an Engine instance
engine = Engine(variables=variables, name="My Simple Rule Engine")

# 3. Add your logical rules and initial evidence
engine.add_rule("x1 = (x2 && x3)")
engine.add_rule("x2 <= (!x3 || !x4)")
engine.add_evidence({"x4": False})

# --- SCENARIO A: Pre-compiling for Repeated Use ---
print("--- SCENARIO A: Pre-compiling for Repeated Predictions ---")

# 4. Compile the rules. This is the recommended approach for multiple queries.
print("\n[Step A.1] Compiling the engine...")
engine.compile()

# 5. Query the compiled "valid set"
print("\n[Step A.2] Inspecting the compiled 'valid set':")
valid_set = engine.valid_set
if valid_set:
    valid_set.print(indent=4)

    # Get the consolidated value of a variable from the compiled set
    print("\nQuerying variable values directly from the 'valid set':")
    x1_value = engine.get_variable_value("x1")
    print(f"  - Consolidated value of 'x1': {x1_value}")
    x2_value = engine.get_variable_value("x2")
    print(f"  - Consolidated value of 'x2': {x2_value}")

# 6. Perform multiple fast predictions using the compiled engine
print("\n[Step A.3] Running multiple fast predictions:")
evidence1 = {"x1": False, "x2": True}
print(f"\n  - Predicting with evidence: {evidence1}")
result1 = engine.predict(evidence1)
if result1:
    x3_val = result1.get_value("x3")
    print(f"    > Prediction for 'x3': {x3_val}")
else:
    print("    > Evidence contradicts the knowledge base.")

evidence2 = {"x1": False, "x3": True}
print(f"\n  - Predicting with evidence: {evidence2}")
result2 = engine.predict(evidence2)
if result2:
    x2_val = result2.get_value("x2")
    print(f"    > Prediction for 'x2': {x2_val}")
else:
    print("    > Evidence contradicts the knowledge base.")

# --- SCENARIO B: On-the-Fly Prediction for a Single Use Case ---
print("\n\n--- SCENARIO B: On-the-Fly Prediction (No Pre-compilation) ---")

# 1. Create a new, un-compiled engine with the same rules
engine_uncompiled = Engine(variables=variables, name="On-the-Fly Engine")
engine_uncompiled.add_rule("x1 = (x2 && x3)")
engine_uncompiled.add_rule("x2 <= (!x3 || !x4)")
engine_uncompiled.add_evidence({"x4": False})

# 2. Perform a prediction directly. This may be faster for a one-off query.
print("\n[Step B.1] Running a single prediction without compiling:")
evidence3 = {"x1": False, "x2": True}
print(f"\n  - Predicting with evidence: {evidence3}")

# The engine multiplies all rules + the evidence in one go.
result3 = engine_uncompiled.predict(evidence3)

# 3. Inspect the result
if result3:
    print("\n  - Resulting StateVector:")
    result3.print(indent=4)
    x3_val_uncompiled = result3.get_value("x3")
    print(f"\n    > Prediction for 'x3': {x3_val_uncompiled}")
else:
    print("    > Evidence contradicts the knowledge base.")
```

## Running Tests

To run the test suite, use pytest through Poetry:

```bash
   poetry run pytest
   ```

## Further Reading & Theoretical Foundations

This library serves as a practical implementation of the concepts described in the following papers.

* [Towards Data Science Article] (Coming Soon) - An article explaining the theory in an accessible way, for which this
  library is a reference implementation.

* State Algebra for Propositional Logic: For a deeper, more theoretical dive into the intricate details of State
  Algebra, see the paper on arXiv: https://arxiv.org/abs/2509.10326
