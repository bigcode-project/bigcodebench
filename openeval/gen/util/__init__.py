import time
import os
import sys
import unittest
import builtins

from copy import deepcopy

from openeval.eval.utils import time_limit, swallow_io


def trusted_exec(code, test_code, entry_point):
    """Execute trusted code in place."""
    # Example of setting environment variables
    # Load other necessary environment settings
    exec_globals = {
        '__builtins__': builtins,
        '__name__': '__main__',
        '__file__': '',  # you might specify the script's intended file path if needed
        '__package__': None,
        '__doc__': None,
        'sys': sys,
        'os': os,
        # Environment variables
        'environ': os.environ,
    }
    
    code = code + "\n" + test_code
    exec(code, exec_globals)
    TestCases = exec_globals['TestCases']
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCases)

    test_result = unittest.TestResult()
    suite.run(test_result)

    # Handle the test results
    if test_result.failures or test_result.errors:
        for test, trace in test_result.failures + test_result.errors:
            print(f"{test.id()}: {trace}")
        raise Exception("Test failed.")
    else:
        print("All tests passed.")


def trusted_check_exec(code, inputs, entry_point):
    """Check trusted_exec success."""
    try:
        with time_limit(seconds=1.0):
            trusted_exec(code, inputs, entry_point)
    except Exception:
        return False
    return True
