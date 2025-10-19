"""
This example demonstrates the basic usage of the vector-logic library and
explains the performance trade-offs of compiling the engine.
"""

from vectorlogic import Engine

# --- Understanding Compilation ---
#
# The vector-logic offers two main approaches for inference, and choosing
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
