import time
import unittest
from copy import deepcopy

from openeval.eval.utils import time_limit, swallow_io


def trusted_exec(code, test_code, entry_point, record_time=False, output_not_none=False):
    """Execute trusted code in place."""
    exec_globals = {}
    code = code + "\n\n" + test_code
    exec(code, exec_globals)
    TestCases = exec_globals['TestCases']
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCases)

    # Measure and print the runtime for each test case
    rtime = []
    test_result = unittest.TestResult()
    for test in suite:
        with swallow_io():
            start = time.time()
            test.run(unittest.TestResult())
            rtime.append(time.time() - start)
    # if test_result.errors or test_result.failures:
    #     for test_case, error_message in test_result.errors:
    #         print(f"Error in {test_case.id()}: {error_message}")
    #     for test_case, failure_message in test_result.failures:
    #         print(f"Failure in {test_case.id()}: {failure_message}")
    # else:
    #     print("All tests passed.")
    if test_result.errors or test_result.failures:
        raise Exception("Test failed.")
        exit(1)
    else:
        print("All tests passed.")
    return rtime


def trusted_check_exec(code, inputs, entry_point):
    """Check trusted_exec success."""
    try:
        with time_limit(seconds=1.0):
            trusted_exec(code, inputs, entry_point)
    except Exception:
        return False
    return True
