import time
from tabnanny import verbose

import numpy as np

from vectorlogic.engine import Engine

_my_spec_ = "Intel_i7-12700H_31GB_Kubuntu_24.04.3_6.8.0-85-generic"


def generate_random_rule(variables_: list[str], num_vars: int, operators_: list[str], random_state) -> str:
    """Generates a random rule string with the given number of variables."""
    rule_vars = random_state.choice(variables_, num_vars).tolist()

    def maybe_negate(var: str) -> str:
        return f"!{var}" if random_state.choice([True, False]) else var

    # Build a simple chained rule, e.g., v1 op v2 op v3 ...
    rule_parts = [maybe_negate(rule_vars[0])]
    for i in range(1, num_vars):
        op = random_state.choice(operators_)
        var_part = maybe_negate(rule_vars[i])
        # Add parentheses randomly to create more complex ASTs
        rule_parts.extend([op, var_part])
        if random_state.random() < 0.3 and i < num_vars - 1:
            rule_so_far = " ".join(rule_parts)
            rule_so_far = f"({rule_so_far})"
            rule_parts = [rule_so_far]

    return " ".join(rule_parts)


def generate_engine(num_rules, num_vars, random_state, verbose=False):
    variables = [f"v{i + 1:02d}" for i in range(num_vars)]
    operators = ["&&", "||", "=>", "<=", "=", "^^"]
    engine = Engine(variables=variables, name="Performance Test Engine")
    for _ in range(num_rules):
        num_rule_vars = random_state.randint(3, 5)  # 3 or 4 variables in the rule
        rule = generate_random_rule(variables, num_rule_vars, operators, random_state)
        engine.add_rule(rule)
    engine._verbose = verbose
    return engine


def measure_compile_time(engine):
    # --- Measure Compilation Time ---
    start_compile_time = time.perf_counter()
    engine.compile()
    end_compile_time = time.perf_counter()
    compile_duration = end_compile_time - start_compile_time

    return compile_duration, engine


def predict_one(compile_=True, verbose=False):
    print()
    print(f"Running predict_one() with compile = {compile_}")
    seed = 425
    num_rules = 60
    num_vars = 80
    num_evidence = 8
    rng = np.random.RandomState()
    rng.seed(seed)

    # generate random rules
    engine = generate_engine(num_rules=num_rules, num_vars=num_vars, random_state=rng, verbose=verbose)

    # generate random evidence
    variables = engine.variables
    evidence = {}
    selected_vars = rng.choice(variables, num_evidence, replace=False)
    for var in selected_vars:
        evidence[var] = rng.choice([True, False])

    start_time = time.perf_counter()

    if compile_:
        start_compile_time = time.perf_counter()
        engine.compile()
        end_compile_time = time.perf_counter()
        compile_duration = end_compile_time - start_compile_time
        print(f"Compilation duration: {compile_duration:.3g} seconds")
        print(f"intermediate_sizes: {engine.intermediate_sizes_stats}")

    start_predict_time = time.perf_counter()
    output = engine.predict(evidence)
    end_predict_time = time.perf_counter()

    print(f"Size of output: {output.size()}")

    predict_duration = end_predict_time - start_predict_time
    overall_time = end_predict_time - start_time
    print(f"Prediction duration: {predict_duration:.3g} seconds")
    print(f"Overall time: {overall_time:.3g} seconds")
    print(f"intermediate_sizes: {engine.intermediate_sizes_stats}")


def run__compile_stats():
    """
    Measures the performance of engine compilation and prediction with a
    set of randomly generated rules.
    """
    # import random

    N = 40  # Number of variables

    repeat = 30
    rng = np.random.RandomState(442)

    time_mean = []
    time_min = []
    time_max = []
    time_std = []
    # M_range = [10, 15, 20, 25, 30]
    # M_range = [15, 20, 25, 30, 35, 40]
    M_range = list(range(30, 50, 5))
    for M in M_range:
        print()
        print(f"M = {M}")

        durations = []
        for i in range(repeat):
            seed = 1000 * M + i
            rng.seed(seed)
            # print(f"\nrepeat {i + 1} / {repeat}")
            engine = generate_engine(num_rules=M, num_vars=N, random_state=rng)
            duration, engine = measure_compile_time(engine)
            if duration > 5:
                print(f"--- Engine Performance Test ---")
                print(f"seed = {seed}")
                print(f"Compiled {M} rules with {N} variables in {duration:.4f} seconds.")
                print(f"Intermediate sizes: {engine._intermediate_sizes}")
            durations.append(duration)

        t = float(np.mean(durations))
        t_std = float(np.std(durations))
        time_mean.append(t)
        time_std.append(t_std)
        time_min.append(float(min(durations)))
        time_max.append(float(max(durations)))

        print()
        print(f"Summary for M = {M}:")
        print("--------------------")
        print(f"time = {t:.4f} +- {t_std:.4f}   [{time_min[-1]:.3g} - {time_max[-1]:.3g}]")

    print()
    print("Summary:")
    print("-------")
    print(f"M_range = {M_range}")
    print(f"time_mean = {np.round(time_mean, 3)}")
    # print(f"time_std = {np.round(time_std, 3)}")
    print(f"time_max = {np.round(time_max, 3)}")

    previous_result = f"""
    [{_my_spec_}]
    ---------------------
    M_range = [30, 35, 40, 45]
    time_mean = [0.096 0.073 0.043 0.016]
    time_max = [0.508 0.905 0.19  0.102]
    """
    print("\nprevious result:")
    print(previous_result)


def run__compile_one(seed=42, num_rules=60, num_vars=80, verbose=False):
    print()
    print("---  Running compile_one() ---")
    # seed = 42
    # seed = 425
    # num_rules = 60
    # num_vars = 80
    print(f"seed = {seed}")
    print(f"num_rules = {num_rules}")
    print(f"num_vars = {num_vars}")

    rng = np.random.RandomState()
    rng.seed(seed)
    # print(f"\nrepeat {i + 1} / {repeat}")
    engine = generate_engine(num_rules=num_rules, num_vars=num_vars, random_state=rng, verbose=verbose)
    duration, engine = measure_compile_time(engine)
    print(f"intermediate_sizes: {engine.intermediate_sizes_stats}")
    print(f"duration = {duration:.3g} ")
    previous_result = """
    seed = 42
    num_rules = 60
    num_vars = 80
    intermediate_sizes: {'num_entries': 59, 'min': 2, 'mean': 361.6, 'rms': 1392.6, 'max': 7232}
    duration = 1.73 
    
    ---------------------
    seed = 425
    num_rules = 60
    num_vars = 80
    intermediate_sizes: {'num_entries': 59, 'min': 1, 'mean': 1463.8, 'rms': 4669.5, 'max': 21312}
    duration = 3.47 
    """
    previous_result = f"""
    [{_my_spec_}]
    ---------------------
    {previous_result}
    """

    print("\nprevious results:")
    print(previous_result)


def run__to_compile_or_not_to_compile(verbose=False):
    predict_one(compile_=True, verbose=verbose)
    predict_one(compile_=False, verbose=verbose)
    previous_result = """
    Running predict_one() with compile = True
    Compilation duration: 2.84 seconds
    intermediate_sizes: {'num_entries': 79, 'min': 1, 'mean': 601.8, 'rms': 2677.1, 'max': 17608}
    Size of output: 1296
    Prediction duration: 0.0309 seconds
    Overall time: 2.87 seconds
    intermediate_sizes: {'num_entries': 1, 'min': 1296, 'mean': 1296.0, 'rms': 1296.0, 'max': 1296}
    
    Running predict_one() with compile = False
    Size of output: 1728
    Prediction duration: 0.342 seconds
    Overall time: 0.342 seconds
    intermediate_sizes: {'num_entries': 898, 'min': 1, 'mean': 8.4, 'rms': 67.1, 'max': 1728}
    
    """
    previous_result = f"""
    [{_my_spec_}]
    ---------------------
    {previous_result}
    """
    print("\nprevious result:")
    print(previous_result)


if __name__ == "__main__":
    # run__compile_one(seed=42, verbose=True)
    # run__compile_one(seed=425, verbose=True)
    # run__compile_one(seed=666, num_rules=80, num_vars=90, verbose=True)
    # run__compile_stats()
    run__to_compile_or_not_to_compile(verbose=True)
