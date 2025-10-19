# vector-logic: A Lightweight Python Rules Engine

A fast, transparent, and extensible Python library for running inference on a generic system of logical rules.

This library is for developers who need a simple and powerful way to manage business logic without the overhead of more
complex systems like BDDs or SAT solvers. It provides an intuitive, algebraic approach to propositional logic, making it
easy to define rules, add evidence, and infer outcomes.

## Why Use vector-logic?

- **Expressive Rule Syntax:** Define complex rules with a natural, human-readable syntax (e.g., (sky_is_grey &&
  humidity_is_high) => it_will_rain).

- **Powerful Inference:** Combine multiple rules and evidence into a single, consolidated knowledge base to make
  predictions and check for contradictions.

- **State Analysis:** Query the state of any variable to determine if it is always true, always false, or can vary based
  on the known rules.

- **Lightweight & Transparent:** A small, focused codebase with minimal dependencies, making it easy to understand,
  extend, and integrate into any project.

## Installation

This project is managed with Poetry. It can be installed from PyPI:

```bash
   pip install vector-logic
   ```

## Quick Start

There is a minimal example of defining rules and making a prediction.

```python
from vectorlogic import Engine

# 1. Define the variables for your system
variables = ["sky_is_grey", "humidity_is_high", "it_will_rain"]

# 2. Create an Engine instance
engine = Engine(variables=variables)

# 3. Add your logical rules
engine.add_rule("sky_is_grey && humidity_is_high => it_will_rain")

# 4. Make a prediction based on new evidence
evidence = {"sky_is_grey": True, "humidity_is_high": True}
result = engine.predict(evidence)

# 5. Check the result
if result:
    rain_prediction = result.get_value("it_will_rain")
    print(f"Will it rain? Prediction: {rain_prediction == 1}")

# Expected Output: Will it rain? Prediction: True
```

## Detailed Example

This example demonstrates more advanced features, including compiling the engine for faster repeated predictions.

```python
from vectorlogic import Engine

# 1. Define variables and create the engine
variables = ["x1", "x2", "x3", "x4"]
engine = Engine(variables=variables, name="My Logic Engine")

# 2. Add rules and initial evidence
engine.add_rule("x1 = (x2 && x3)")
engine.add_rule("x2 <= (!x3 || !x4)")
engine.add_evidence({"x4": False})

# 3. Compile the rules into a single 'valid set'
# (See the performance section below for when to use this)
engine.compile()

# 4. Inspect the compiled state
print("--- Compiled Valid Set ---")
engine.valid_set.print(indent=4)

# 5. Query the compiled knowledge base directly
x2_value = engine.get_variable_value("x2")
print(f"\nConsolidated value of 'x2' in the valid set: {x2_value}")

# 6. Run a fast prediction using the compiled engine
print("\n--- Prediction ---")
evidence = {"x1": False, "x2": True}
print(f"Predicting with evidence: {evidence}")
result = engine.predict(evidence)

if result:
    x3_value = result.get_value("x3")
    print(f"Inferred value of 'x3': {x3_value}")
else:
    print("Evidence contradicts the knowledge base.")
```

## Performance: To Compile or Not to Compile?

The `vector-logic` engine offers two approaches for inference, and the best choice depends on your use case.

1. **Pre-compiling for Repeated Use** (`engine.compile()`)

- **What it does:** Multiplies all added rules together to create a single, optimised `StateVector` called the "valid
  set".

- **When to use it:** When you need to run multiple predictions against the same set of rules.

- **Trade-off:** The initial compilation can be slow if the valid set is very large, but subsequent `predict()` calls
  will
  be extremely fast because they only need to multiply with this single, pre-computed valid set.

2. **On-the-Fly Prediction**

- **What it does:** Multiplies all rules from scratch every time you call `.predict()`, including the evidence for that
  specific prediction.

- **When to use it:** When you need to run one or a few predictions.

- **Trade-off:** This can be faster for a single query because a restrictive piece of evidence can cause the
  intermediate `StateVectors` to become very small, avoiding the creation of a potentially huge valid set.

> **A Note on Performance:** The efficiency of this algorithm relies on heuristic optimisations for the order of rule
> multiplication. This implementation includes a basic but effective heuristic that performs well for systems with up to
> ~100 variables and a similar number of rules. For more challenging tasks, performance can be improved by implementing
> more advanced optimisation heuristics.

## Installation from source

This project uses Poetry for dependency management.

1. Install Poetry:
   Follow the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

2. Clone the repository:

    ```bash
    git clone https://github.com/dmitry-lesnik/vector-logic.git
    cd vector-logic
    ```

3. Install dependencies:
    ```bash
    poetry install
    ```

## Running Tests

To run the test suite, use pytest through Poetry:

```bash
   poetry install --with dev
   poetry run pytest
   ```

## Further Reading & Theoretical Foundations

This library serves as a practical implementation of the concepts described in the following papers.

* [Towards Data Science Article] (Coming Soon) - An article explaining the theory in an accessible way, for which this
  library is a reference implementation.

* **State Algebra for Propositional Logic:** For a deeper, more theoretical dive into the intricate details of State
  Algebra, see the paper on arXiv: https://arxiv.org/abs/2509.10326

