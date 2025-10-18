"""
An example that replicates the "Importation-Exportation" logical inference
proof from Appendix A of the State Algebra paper.
"""

# To run this example from the root of the project, you might need to
# add the project's root directory to the Python path, e.g., by running:
# export PYTHONPATH=$(pwd)
# python examples/importation_exportation_example.py

from rule_engine import Engine


def importation_exportation_proof():
    """
    This example proves the "Importation-Exportation" rule of propositional logic:
    (E1 -> (E2 -> E3)) is logically equivalent to ((E1 && E2) -> E3)

    To do this, we use supplementary variables to break down the problem,
    exactly as described in the paper.
    """
    print("--- Running Importation-Exportation Rule Proof ---")

    # 1. Define the variables. E1, E2, E3 are the core variables.
    #    E4, E5, E6, E7 are supplementary variables to illustrate the handling of nested rules.
    #    E8 is the final indicator variable for the proposition we want to prove.
    variables = [f"E{i}" for i in range(1, 9)]

    # 2. Create the Engine instance.
    engine = Engine(variables=variables, name="Logic Proof Engine")

    # 3. Add the definitions of the supplementary variables as rules.
    print("\n[Step 1] Adding definitions for supplementary variables...")
    engine.add_rule("E4 = (E2 => E3)")  # Premise part 1
    engine.add_rule("E5 = (E1 => E4)")  # Premise (LHS of the main formula)
    # Complex rules can also be added directly
    engine.add_rule("E7 = ((E1 && E2) => E3)")  # Consequent (RHS of the main formula)

    # 4. Add the final proposition we want to prove.
    #    We want to prove that the premise implies the consequent, which is E5 => E7.
    #    We create an indicator variable E8 for this proposition. If E8 is always
    #    True (value 1), the proposition is a tautology.
    print("\n[Step 2] Adding the final proposition to prove (E5 => E7)...")
    engine.add_rule("E8 = (E5 => E7)")

    # Let's see the initial setup
    engine.print()

    # 5. Compile the entire knowledge base.
    #    The engine will multiply all the state vectors from the rules above.
    print("\n[Step 3] Compiling the entire knowledge base...")
    engine.compile()

    print("\n--- Compiled Knowledge Base ---")
    engine.print()

    # Check for contradictions in the rule set
    if engine.is_contradiction():
        print("\nWarning: The rule set contains a contradiction!")
        return
    print("No logical contradictions found.")

    # 6. Check the value of the indicator variable E8.
    print("\n[Step 4] Checking the value of the indicator variable E8...")
    e8_value = engine.get_variable_value("E8")

    print(f"\nThe consolidated value of E8 is: {e8_value}")

    if e8_value == 1:
        print("\nConclusion: E8 is identically True. The proposition (E5 => E7) is a tautology.")
        print("This proves that (E1 -> (E2 -> E3)) implies ((E1 && E2) -> E3).")
    else:
        print("\n!!!  Aaaaaaa: E8 is not identically True. The proof has failed.")
    print("--------------------------")

    # We can also check the stronger claim from the paper: are E5 and E7 equivalent?
    print("\n[Step 5] Checking for equivalence (E5 = E7)...")
    equivalence_engine = Engine(variables=variables)
    equivalence_engine.add_rule("E5 = (E1 => (E2 => E3))")
    equivalence_engine.add_rule("E7 = ((E1 && E2) => E3)")
    # Add the equivalence check
    equivalence_engine.add_rule("E8 = (E5 = E7)")
    equivalence_engine.compile()
    e8_equivalence_value = equivalence_engine.get_variable_value("E8")

    if e8_equivalence_value == 1:
        print("\nThe value of E8 for (E5 = E7) is 1. This confirms they are logically equivalent.")
    else:
        print("\n!!!  The value of E8 for (E5 = E7) is not 1. They are not equivalent.")
    print("--------------------------")

    # We can also test a prediction that should result in a contradiction
    print("\n[Step 6] Testing a prediction that causes a contradiction...")
    # The rules imply E5 = E7. Let's add evidence that E5 is true and E7 is false.
    # This should result in a contradiction.
    contradictory_evidence = {"E5": True, "E7": False}
    print(f"Predicting with evidence: {contradictory_evidence}")
    prediction_result = equivalence_engine.predict(contradictory_evidence)

    if prediction_result.is_contradiction():
        print("Prediction resulted in a contradiction, as expected.")


if __name__ == "__main__":
    importation_exportation_proof()
