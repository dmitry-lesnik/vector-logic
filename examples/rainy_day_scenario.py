"""
A complete example of using the rule_engine library to perform logical inference.
"""

# To run this example from the root of the project, you might need to
# add the project's root directory to the Python path, e.g., by running:
# export PYTHONPATH=$(pwd)
# python examples/inference_example.py

from rule_engine import Engine


def rainy_day_scenario():
    """
    Demonstrates a simple inference scenario about whether to take an umbrella.
    """
    print("--- Running Rainy Day Inference Scenario ---")

    # 1. Define the variables involved in our logical world.
    variables = [
        "sky_is_grey",
        "humidity_is_high",
        "wind_is_strong",
        "it_will_rain",
        "take_umbrella",
    ]

    # 2. Create an instance of the Engine.
    engine = Engine(variables=variables, name="Rainy Day Advisor")

    # 3. Add the knowledge base as a series of logical rules.
    print("\n[Step 1] Adding logical rules to the knowledge base...")
    engine.add_rule("sky_is_grey && humidity_is_high => it_will_rain")
    engine.add_rule("it_will_rain => take_umbrella")
    engine.add_rule("wind_is_strong = !take_umbrella")  # A constraint

    # 4. Compile all the rules into a single, combined StateVector.
    # This represents the complete, unified knowledge of the system.
    print("\n[Step 2] Compiling the knowledge base...")
    engine.compile()

    # Let's print the engine's state to see the compiled knowledge.
    engine.print()

    # 5. Perform inference by providing evidence and making predictions.

    # --- Scenario A: Clear conditions for rain ---
    print("\n[Step 3] Performing inference with new evidence...")
    print("\n--- Scenario A ---")
    evidence_a = {"sky_is_grey": True, "humidity_is_high": True}
    print(f"Evidence: The sky is grey AND humidity is high.")

    # The predict() method combines the evidence with the compiled knowledge
    # and returns a new InferenceResult object.
    result_a = engine.predict(evidence_a)
    print("\nResulting StateVector for this scenario:")
    result_a.print(max_index=len(variables), indent=4)

    # We can now query this result by variable name.
    if result_a:
        rain_prediction = result_a.get_value("it_will_rain")
        umbrella_prediction = result_a.get_value("take_umbrella")

        print(f"Prediction for 'it_will_rain': {rain_prediction} (1=True, 0=False, -1=Unknown)")
        print(f"Prediction for 'take_umbrella': {umbrella_prediction}")
        # Expected outcome: it_will_rain is True (1), take_umbrella is True (1).
    else:
        print("Contradictory evidence !!!")

    # --- Scenario B: Conflicting conditions ---
    print("\n--- Scenario B ---")
    evidence_b = {
        "sky_is_grey": True,
        "humidity_is_high": True,
        "wind_is_strong": True,
    }
    print("Evidence: The sky is grey, humidity is high, AND the wind is strong.")

    result_b = engine.predict(evidence_b)
    print("\nResulting StateVector for this scenario:")
    result_b.print(max_index=len(variables), indent=4)

    # Expected outcome: The rules conflict! `it_will_rain` implies `take_umbrella`,
    # but `wind_is_strong` implies `!take_umbrella`. This results in a null
    # or empty StateVector, correctly showing a logical contradiction.
    if result_b:
        rain_prediction_b = result_b.get_value("it_will_rain")
        umbrella_prediction_b = result_b.get_value("take_umbrella")

        print(f"Prediction for 'it_will_rain': {rain_prediction_b}")
        print(f"Prediction for 'take_umbrella': {umbrella_prediction_b}")
    else:
        print("Logical contradiction !!!")

    # --- Scenario C: Querying the base knowledge ---
    print("\n[Step 4] Querying the compiled knowledge without new evidence...")
    print("\n--- Scenario C ---")
    # We can also ask questions about the base knowledge without any new evidence.
    # For example, what does the engine know about `take_umbrella` if we know
    # that the wind is strong?
    result_c = engine.predict({"wind_is_strong": True})
    umbrella_if_windy = result_c.get_value("take_umbrella")
    print(f"If we know the wind is strong, the value of 'take_umbrella' is: {umbrella_if_windy}")
    # Expected outcome: take_umbrella is False (0).


if __name__ == "__main__":
    rainy_day_scenario()
