#!/usr/bin/env python3

import itertools
import subprocess
import sys

from subprocess import DEVNULL


default_config = {
    "PERPENDICULAR_RAYS": 1,
    "COMPACTION": 1,
    "FORCE_SINGLE_ANYHIT": 1,
    "START_RAY_AT_ZERO": 0,
    "LARGE_KEYS": 0,
    "PERFORM_UPDATES": 0,
    "LEAVE_GAPS_FOR_MISSES": 0,

    "PRIMITIVE": 0,

    "INSERT_SORTED": 0,
    "PROBE_SORTED": 0,

    "EXPONENT_BIAS": 0,
    "NUM_BUILD_KEYS_LOG": 25,
    "NUM_PROBE_KEYS_LOG": 27,
    "RANGE_QUERY_HIT_COUNT_LOG": 0,
    "NUM_UPDATES_LOG": 0,
    "MISS_PERCENTAGE": 0,
    "OUT_OF_RANGE_PERCENTAGE": 0,
    "KEY_STRIDE_LOG": 0,

    "INT_TO_FLOAT_CONVERSION_MODE": 3,
}

experiments = {
    "int-to-ray": {
        "NUM_BUILD_KEYS_LOG": [21, 22, 23, 24, 25, 26],
        "PERPENDICULAR_RAYS": [0, 1],
        "START_RAY_AT_ZERO": [1],
        "INT_TO_FLOAT_CONVERSION_MODE": [0, 1, 2, 3],
    },
    "primitive": {
        "NUM_BUILD_KEYS_LOG": [21, 22, 23, 24, 25, 26],
        "PERPENDICULAR_RAYS": [0, 1],
        "START_RAY_AT_ZERO": [0, 1],
        "PROBE_SORTED": [0, 1],
        "COMPACTION": [0, 1],
        "PRIMITIVE": [0, 1, 2],
    },
    "exponent-shift": {
        "NUM_BUILD_KEYS_LOG": [21, 22, 23, 24, 25, 26],
        "START_RAY_AT_ZERO": [1],
        "INT_TO_FLOAT_CONVERSION_MODE": [0, 1, 2, 3],
        "EXPONENT_BIAS": [-20, -10, 0, 10, 20],
    },
    "stride": {
        "NUM_BUILD_KEYS_LOG": [21, 22, 23, 24, 25, 26],
        "START_RAY_AT_ZERO": [1],
        "INT_TO_FLOAT_CONVERSION_MODE": [0, 1, 2, 3],
        "KEY_STRIDE_LOG": [0, 1, 2, 3],
    },
    "updates": {
        "NUM_BUILD_KEYS_LOG": [25, 26],
        "PERFORM_UPDATES": [1],
        "NUM_UPDATES_LOG": [0, 2, 4, 6, 8, 10, 12],
    },
    "key-size": {
        "NUM_BUILD_KEYS_LOG": [25, 26],
        "LARGE_KEYS": [0, 1],
    },
    "ordering": {
        "NUM_BUILD_KEYS_LOG": [25, 26],
        "INSERT_SORTED": [-1, 0, 1],
        "PROBE_SORTED": [-1, 0, 1, 2],
    },
    "range-query-start-ray-at-zero": {
        "NUM_BUILD_KEYS_LOG": [25],
        "RANGE_QUERY_HIT_COUNT_LOG": [0, 2, 4, 6, 8, 10, 12],
        "START_RAY_AT_ZERO": [0, 1],
    },
    "range-query-ordering": {
        "NUM_BUILD_KEYS_LOG": [25],
        "RANGE_QUERY_HIT_COUNT_LOG": [0, 2, 4, 6, 8, 10, 12],
        "PROBE_SORTED": [0, 2],
    },
    "range-query-conversion-mode": {
        "NUM_BUILD_KEYS_LOG": [25],
        "RANGE_QUERY_HIT_COUNT_LOG": [0, 2, 4, 6, 8],
        "START_RAY_AT_ZERO": [1],
        "INT_TO_FLOAT_CONVERSION_MODE": [0, 1, 2, 3],
    },
#    "miss": {
#        "NUM_BUILD_KEYS_LOG": [25, 26],
#        "PROBE_SORTED": [0, 2],
#        "LEAVE_GAPS_FOR_MISSES": [1],
#        "MISS_PERCENTAGE": [0, 25, 50, 90, 99, 100],
#    },
#    "miss-out-of-range": {
#        "NUM_BUILD_KEYS_LOG": [25, 26],
#        "PROBE_SORTED": [0, 2],
#        "LEAVE_GAPS_FOR_MISSES": [1],
#        "OUT_OF_RANGE_PERCENTAGE": [0, 25, 50, 90, 99, 100],
#    },
}

if len(sys.argv) > 1:
    experiments = {name: experiments[name] for name in sys.argv[1:]}

for description, configurations in experiments.items():

    # delete result file
    output_file = f"{description}.csv"
    subprocess.run(["rm", output_file])

    with open(output_file, "w+") as output_file:
        replicate_key = [[(key, value) for value in values] for key, values in configurations.items()]
        all_combinations = [dict(test_config) for test_config in itertools.product(*replicate_key)]

        for test_config in all_combinations:
            assert default_config.keys() >= test_config.keys()
            modified_config = {**default_config, **test_config}

            generated_header = [f"#define {key} {value}\n" for key, value in modified_config.items()]
            print("".join(generated_header), file=sys.stderr, flush=True)

            with open(f"../src/test_configuration_override.h", "w") as header_file:
                header_file.writelines(generated_header)
            # build and run
            subprocess.run(["make", "-B", "-j", "16"], stdout=DEVNULL, stderr=DEVNULL).check_returncode()
            try:
                subprocess.run(["./run_experiment"], stdout=output_file, timeout=300).check_returncode()
            except subprocess.TimeoutExpired:
                pass

            print("", file=sys.stderr, flush=True)
