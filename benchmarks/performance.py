import time

import numpy as np

from vectorlogic.engine import Engine


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


def generate_engine(num_rules, num_vars, random_state):
    variables = [f"v{i + 1:02d}" for i in range(num_vars)]
    operators = ["&&", "||", "=>", "<=", "=", "^^"]
    engine = Engine(variables=variables, name="Performance Test Engine")
    for _ in range(num_rules):
        num_rule_vars = random_state.randint(3, 5)  # 3 or 4 variables in the rule
        rule = generate_random_rule(variables, num_rule_vars, operators, random_state)
        engine.add_rule(rule)
    return engine


def measure_compile_time(engine):
    # --- Measure Compilation Time ---
    start_compile_time = time.perf_counter()
    engine.compile()
    end_compile_time = time.perf_counter()
    compile_duration = end_compile_time - start_compile_time

    return compile_duration, engine


def predict_one(compile_=True):
    print()
    print(f"Running predict_one() with compile = {compile_}")
    seed = 425
    num_rules = 60
    num_vars = 80
    num_evidence = 8
    rng = np.random.RandomState()
    rng.seed(seed)

    # generate random rules
    engine = generate_engine(num_rules=num_rules, num_vars=num_vars, random_state=rng)

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
        print(f"Compilation duration: {compile_duration:.4f} seconds")
        print(f"intermediate_sizes: {engine.intermediate_sizes}")

    start_predict_time = time.perf_counter()
    output = engine.predict(evidence)
    end_predict_time = time.perf_counter()

    print(f"Size of output: {output.size()}")

    predict_duration = end_predict_time - start_predict_time
    overall_time = end_predict_time - start_time
    print(f"Prediction duration: {predict_duration:.4f} seconds")
    print(f"Overall time: {overall_time:.4f} seconds")
    print(f"intermediate_sizes: {engine.intermediate_sizes}")


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

    previous_result = """
    [Intel_i7-12700H_31GB_Kubuntu_24.04.3_6.8.0-85-generic]
    ---------------------
    M_range = [30, 35, 40, 45]
    time_mean = [0.096 0.073 0.043 0.016]
    time_max = [0.508 0.905 0.19  0.102]
    """
    print("\nprevious result:")
    print(previous_result)


def run__compile_one():
    print()
    print("---  Running compile_one() ---")
    seed = 425
    num_rules = 60
    num_vars = 80
    print(f"seed = {seed}")
    print(f"num_rules = {num_rules}")
    print(f"num_vars = {num_vars}")

    rng = np.random.RandomState()
    rng.seed(seed)
    # print(f"\nrepeat {i + 1} / {repeat}")
    engine = generate_engine(num_rules=num_rules, num_vars=num_vars, random_state=rng)
    duration, engine = measure_compile_time(engine)
    print("--- State Vector sizes during compilation:")
    print(engine.intermediate_sizes)
    print(f"duration = {duration:.3g} ")
    previous_result = """
    [Intel_i7-12700H_31GB_Kubuntu_24.04.3_6.8.0-85-generic]
    ---------------------
    seed = 42
    num_rules = 60
    num_vars = 80
    --- State Vector sizes during compilation:
    [6, 4, 5, 4, 7, 6, 8, 5, 11, 2, 4, 4, 8, 9, 6, 12, 6, 6, 3, 6, 5, 6, 3, 6, 6, 6, 5, 13, 9, 8, 21, 12, 5, 7, 28, 130, 4, 9, 34, 32, 2, 2, 38, 56, 24, 18, 447, 132, 140, 508, 24, 192, 1959, 110, 7036, 12, 2864, 66, 7232]
    duration = 1.67 
    
    ---------------------
    seed = 425
    num_rules = 60
    num_vars = 80
    --- State Vector sizes during compilation:
    [2, 1, 3, 6, 6, 5, 5, 2, 1, 7, 16, 3, 22, 4, 3, 6, 4, 8, 2, 5, 6, 3, 6, 6, 4, 6, 5, 3, 10, 10, 112, 12, 9, 10, 9, 12, 24, 86, 1069, 10, 16, 66, 11, 135, 4, 22, 12, 42, 2984, 21312, 21012, 6970, 24, 5550, 3722, 4006, 928, 1662, 16362]
    duration = 3.43 
    """
    print("\nprevious results:")
    print(previous_result)


def run__to_compile_or_not_to_compile():
    predict_one(compile_=True)
    predict_one(compile_=False)
    previous_result = """
    [Intel_i7-12700H_31GB_Kubuntu_24.04.3_6.8.0-85-generic]
    ---------------------
    Running predict_one() with compile = True
    Compilation duration: 3.4746 seconds
    intermediate_sizes: [2, 1, 3, 6, 6, 5, 5, 2, 1, 7, 16, 3, 22, 4, 3, 6, 4, 8, 2, 5, 6, 3, 6, 6, 4, 6, 5, 3, 10, 10, 112, 12, 9, 10, 9, 12, 24, 86, 1069, 10, 16, 66, 11, 135, 4, 22, 12, 42, 2984, 21312, 21012, 6970, 24, 5550, 3722, 4006, 928, 1662, 16362]
    Size of output: 1728
    Prediction duration: 0.0518 seconds
    Overall time: 3.5264 seconds
    intermediate_sizes: [1728]
    
    Running predict_one() with compile = False
    Size of output: 1728
    Prediction duration: 1.2630 seconds
    Overall time: 1.2630 seconds
    intermediate_sizes: [2, 1, 3, 6, 6, 5, 5, 2, 1, 7, 16, 3, 22, 4, 3, 6, 4, 8, 3, 5, 5, 7, 3, 1, 3, 7, 6, 1, 4, 6, 2, 7, 30, 4, 9, 16, 10, 112, 9, 533, 24, 30, 64, 5172, 7, 10, 16, 66, 20, 11, 64, 8032, 624, 22, 864, 18, 356, 121, 6336, 1728]
    """
    print("\nprevious result:")
    print(previous_result)


if __name__ == "__main__":
    # run__compile_one()
    run__compile_stats()
    # run__to_compile_or_not_to_compile()
