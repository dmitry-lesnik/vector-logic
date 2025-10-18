"""
Tests for the Engine class.
"""

import pytest

from rule_engine.engine import Engine
from rule_engine.state_vector import StateVector
from rule_engine.t_object import TObject


@pytest.fixture
def base_engine():
    """Provides a basic engine with three variables."""
    return Engine(variables=["x1", "x2", "x3"], name="Test Engine")


def test_engine_initialization(base_engine):
    """Tests the basic initialization of the Engine."""
    assert base_engine._name == "Test Engine"
    assert base_engine._variables == ["x1", "x2", "x3"]
    assert base_engine._variable_map == {"x1": 1, "x2": 2, "x3": 3}
    assert base_engine.rules == []
    assert base_engine.state_vectors == []


def test_engine_initialization_minimal():
    """Tests the minimal initialization of the Engine class with valid variables."""
    variables = ["a", "b", "c", "_start", "var_1"]
    engine = Engine(variables=variables)

    # Variables are sorted internally now
    expected_vars = sorted(variables)
    assert engine._variables == expected_vars
    assert engine._name is None
    assert engine._uncompiled_rules == []
    assert engine._state_vectors == []
    assert engine._variable_map == {"_start": 1, "a": 2, "b": 3, "c": 4, "var_1": 5}


def test_engine_initialization_full():
    """Tests the full initialization of the Engine class with all optional arguments."""
    variables = ["x", "y", "z_val"]
    rules = ["x => y", "y && z_val"]
    name = "MyTestEngine"
    engine = Engine(variables=variables, name=name, rules=rules)

    expected_vars = sorted(variables)
    assert engine._variables == expected_vars
    assert engine._name == name
    assert engine._uncompiled_rules == rules
    assert len(engine._state_vectors) == 2
    assert engine._variable_map["x"] == 1
    assert engine._variable_map["y"] == 2
    assert engine._variable_map["z_val"] == 3


def test_engine_handles_duplicate_variables():
    """
    Tests that the Engine correctly handles duplicate variable names,
    resulting in a unique, sorted list.
    """
    variables = ["b", "c", "a", "c", "b"]
    engine = Engine(variables=variables)

    expected_unique_vars = ["a", "b", "c"]
    assert engine._variables == expected_unique_vars
    assert engine._variable_map == {"a": 1, "b": 2, "c": 3}


def test_variable_uniqueness_and_sorting():
    """Tests that the variable list is made unique and is sorted."""
    engine = Engine(variables=["x3", "x1", "x2", "x1"])
    assert engine._variables == ["x1", "x2", "x3"]
    assert engine._variable_map == {"x1": 1, "x2": 2, "x3": 3}


def test_engine_invalid_variables():
    """Tests that Engine raises ValueError for a variable starting with a number."""
    with pytest.raises(ValueError, match="'1b' is not conformal"):
        Engine(variables=["a", "1b", "c"])

    """Tests that Engine raises ValueError for a variable with special characters."""
    with pytest.raises(ValueError, match="'b-c' is not conformal"):
        Engine(variables=["a", "b-c", "d"])

    with pytest.raises(ValueError, match="'b\\$c' is not conformal"):
        Engine(variables=["a", "b$c", "d"])

    with pytest.raises(ValueError, match="Variable name '1x' is not conformal."):
        Engine(variables=["x1", "1x"])

    """Tests that Engine raises ValueError for an empty variable name."""
    with pytest.raises(ValueError, match="'' is not conformal"):
        Engine(variables=["a", "", "c"])


def test_engine_initialization_with_rules():
    """Tests initializing the Engine with a list of rules."""
    engine = Engine(variables=["x1", "x2"], rules=["x1 => x2"])
    assert len(engine.rules) == 1
    assert len(engine.state_vectors) == 1
    expected_sv = StateVector([TObject(zeros={1}), TObject(ones={1, 2})])
    assert engine.state_vectors[0] == expected_sv


def test_add_rule(base_engine):
    """Tests the add_rule method."""
    base_engine.add_rule("x1 || x2")
    assert len(base_engine.rules) == 1
    assert base_engine.rules[0] == "x1 || x2"
    assert len(base_engine.state_vectors) == 1
    expected_sv = StateVector([TObject(ones={1}), TObject(zeros={1}, ones={2})])
    assert base_engine.state_vectors[0] == expected_sv


def test_add_rule_complex():
    """
    Tests adding a complex rule, which should be flattened and converted.
    Rule: (x1 && x2) => x3
    """
    engine = Engine(variables=["x1", "x2", "x3"])
    engine.add_rule("(x1 && x2) => x3")

    assert len(engine._uncompiled_rules) == 1
    assert len(engine._state_vectors) == 1

    expected_sv = StateVector([TObject(ones={1, 2, 3}), TObject(zeros={1}), TObject(ones={1}, zeros={2})])

    assert engine._state_vectors[0] == expected_sv


def test_add_evidence(base_engine):
    """Tests the add_evidence method."""
    evidence = {"x1": True, "x3": False}
    base_engine.add_evidence(evidence)

    assert len(base_engine.rules) == 1
    assert len(base_engine.state_vectors) == 1
    assert base_engine.rules[0] == f"evidence: {evidence}"

    expected_sv = StateVector([TObject(ones={1}, zeros={3})])
    assert base_engine.state_vectors[0] == expected_sv


def test_add_state_vector(base_engine):
    """Tests the add_evidence method."""
    sv = StateVector([TObject(ones={1}, zeros={3})])
    base_engine.add_state_vector(sv)

    assert len(base_engine.rules) == 1
    assert len(base_engine.state_vectors) == 1
    assert base_engine.rules[0] == f"custom state vector"

    expected_sv = StateVector([TObject(ones={1}, zeros={3})])
    assert base_engine.state_vectors[0] == expected_sv


def test_print_method(base_engine, capsys):
    """Tests the print method by capturing stdout."""
    base_engine.add_rule("x1 => x2")
    base_engine.add_evidence({"x3": True})
    base_engine.print()

    captured = capsys.readouterr()
    output = captured.out

    # Check for key components in the output
    assert "====== Engine: Test Engine ======" in output
    assert "Variables: ['x1', 'x2', 'x3']" in output
    assert "1. Rule:  x1 => x2" in output
    assert "2. Rule:  evidence: {'x3': True}" in output
    # Check for a piece of the formatted StateVector for the first rule
    assert "    1 1 -\n    0 - -" in output
    # Check for a piece of the formatted StateVector for the evidence
    assert "    - - 1" in output


def test_compilation_with_trivial(base_engine, capsys):
    base_engine.add_rule("x1 => x2")
    base_engine.add_rule("x2 => x3")
    base_engine.print()
    base_engine.compile()
    base_engine.print()

    engine = Engine(variables=["x1", "x2", "x3"])
    engine.add_rule("x1 => x2")
    engine.add_rule("x2 => x3")
    engine.add_state_vector(StateVector([TObject()]))
    engine.compile()
    engine.print(debug_info=True)
    assert engine.valid_set == base_engine.valid_set
    # ---- comment out to see the output ------
    capsys.readouterr()


def test_compilation_with_null(base_engine, capsys):
    base_engine.add_rule("x1 => x2")
    base_engine.add_rule("x2 => x3")
    base_engine.add_state_vector(StateVector())
    with pytest.raises(AttributeError):
        _ = base_engine.valid_set
    base_engine.print()
    base_engine.compile()
    base_engine.print()
    assert base_engine.is_contradiction()
    # ---- comment out to see the output ------
    capsys.readouterr()


def test_workflow(base_engine):
    # 1. Setup the Engine
    engine = Engine(variables=["x1", "x2", "x3"])
    engine.add_rule("x1 => x2")
    engine.add_rule("x2 => x3")

    # 2. Compile the knowledge base
    engine.compile()
    # engine.print()  # Shows the combined state vector

    # 3. Scenario 1: Query the compiled state
    value_of_x2 = engine.get_variable_value("x2")
    # print(f"The value of x2 is: {value_of_x2}")

    # 4. Scenario 2: Test multiple pieces of evidence
    evidence1 = {"x1": True}
    result1 = engine.predict(evidence1)
    # print(f"With evidence {evidence1}, the value of x3 is: {result1.get_value('x3')}")
    assert result1.get_value("x3") == 1

    evidence2 = {"x3": False}
    result2 = engine.predict(evidence2)
    # print(f"With evidence {evidence2}, the value of x1 is: {result2.get_value('x1')}")
    assert result2.get_value("x1") == 0

    evidence3 = {"x2": True, "x3": False}
    result2 = engine.predict(evidence3)
    # print(f"With evidence {evidence3}, the state vector is: {result2.state_vector.to_string()}")
    assert not result2


def test_compile_lifecycle(capsys):
    """
    Tests the compilation lifecycle: compiling, adding more rules, and
    re-compiling to ensure knowledge is cumulative.
    """
    print()
    engine = Engine(variables=["x1", "x2", "x3"])
    engine.print()

    # 1. Initial state: Not compiled, no valid_set
    assert not engine._is_compiled
    assert engine._valid_set is None

    # 2. Add one rule and compile
    engine.add_rule("x1 => x2")
    engine.print()
    engine.compile()
    engine.print()

    # 3. State after first compile: compiled, lists are managed
    assert engine._is_compiled
    assert not engine._uncompiled_rules
    assert not engine._state_vectors
    assert len(engine._compiled_rules) == 1
    expected_sv1 = StateVector([TObject(zeros={1}), TObject(ones={1, 2})])
    assert engine.valid_set == expected_sv1

    # 4. Add another rule. This invalidates the compiled state.
    engine.add_rule("x2 => x3")
    engine.print()
    assert not engine._is_compiled
    assert len(engine._uncompiled_rules) == 1
    assert len(engine._state_vectors) == 1
    assert len(engine._compiled_rules) == 1  # Unchanged from last compile

    # 5. Re-compile. The new rule is multiplied with the previous valid_set.
    engine.compile()
    engine.print()
    assert engine._is_compiled
    assert not engine._uncompiled_rules
    assert not engine._state_vectors
    assert len(engine._compiled_rules) == 2

    # The new valid_set is the product of the first two rules.
    # (x1 => x2) * (x2 => x3) which simplifies as tested elsewhere.
    final_expected_sv = StateVector([TObject(zeros={1, 2}), TObject(ones={2, 3})])
    assert engine.valid_set == final_expected_sv

    # 6. Verify the cumulative knowledge using a prediction
    # The combined rules imply that if x1 is true, x3 must be true.
    result = engine.predict({"x1": True})
    assert result.get_value("x3") == 1

    # ---- comment out to see the output ------
    capsys.readouterr()


# =================================================================


def test_predict_consistency_before_and_after_compile(base_engine):
    """
    Tests that predict() returns the same result regardless of compilation status.
    """
    base_engine.add_rule("x1 => x2")
    base_engine.add_rule("x2 => x3")
    evidence = {"x1": True}

    # 1. Predict before compiling
    result_before_compile = base_engine.predict(evidence)

    # 2. Compile the engine
    base_engine.compile()
    assert base_engine.compiled

    # 3. Predict after compiling
    result_after_compile = base_engine.predict(evidence)

    # 4. The results should be identical
    assert not result_before_compile.is_contradiction()
    assert result_before_compile.state_vector == result_after_compile.state_vector

    # 5. Verify the logical outcome is correct in both cases
    expected_sv = StateVector([TObject(ones={1, 2, 3})])
    assert result_before_compile.state_vector == expected_sv
    assert result_after_compile.state_vector == expected_sv


def test_workflow_with_explicit_compilation(base_engine):
    """
    Tests a standard user workflow with explicit compilation and prediction.
    """
    # 1. Setup the Engine
    base_engine.add_rule("x1 => x2")
    base_engine.add_rule("x2 => x3")

    # 2. Compile the knowledge base explicitly
    base_engine.compile()
    assert base_engine.compiled

    # 3. Query the compiled state
    # After compiling x1=>x2 and x2=>x3, if x1 is true, x3 must be true.
    # But without evidence, x1, x2, and x3 are all undetermined.
    assert base_engine.get_variable_value("x1") == -1
    assert base_engine.get_variable_value("x2") == -1
    assert base_engine.get_variable_value("x3") == -1

    # 4. Use predict with evidence
    evidence1 = {"x1": True}
    result1 = base_engine.predict(evidence1)
    assert result1.get_value("x3") == 1

    evidence2 = {"x3": False}
    result2 = base_engine.predict(evidence2)
    assert result2.get_value("x1") == 0


def test_get_value_on_uncompiled_engine(base_engine):
    """
    Tests that get_variable_value raises an AttributeError if the engine is not compiled.
    """
    base_engine.add_rule("x1 = x2")
    with pytest.raises(AttributeError):
        base_engine.get_variable_value("x1")


def test_compilation_with_trivial_sv(base_engine):
    """
    Tests that compiling with a trivial SV doesn't change the result.
    """
    base_engine.add_rule("x1 => x2")
    base_engine.compile()
    expected_valid_set = base_engine.valid_set

    # Create a second engine with an extra trivial rule
    engine2 = Engine(variables=["x1", "x2", "x3"])
    engine2.add_rule("x1 => x2")
    engine2.add_state_vector(StateVector([TObject()]))  # Add trivial vector
    engine2.compile()

    assert engine2.valid_set == expected_valid_set


def test_compilation_with_contradictory_sv(base_engine):
    """
    Tests that compiling with a contradictory (empty) SV results in a contradiction.
    """
    base_engine.add_rule("x1 => x2")
    base_engine.add_state_vector(StateVector())  # Add contradictory vector
    base_engine.compile()

    assert base_engine.is_contradiction()
