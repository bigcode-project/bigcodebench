import hashlib
import json
import os
from typing import Dict

from wildcode.data.utils import (
    CACHE_DIR,
    completeness_check,
    get_dataset_metadata,
    make_cache,
    stream_jsonl,
)

HUMANEVAL_OVERRIDE_PATH = os.environ.get("HUMANEVAL_OVERRIDE_PATH", None)


def get_wild_code_bench() -> Dict[str, Dict]:
    """Get WildCodeBench from OpenAI's github repo and return as a list of parsed dicts.

    Returns:
        List[Dict[str, str]]: List of dicts with keys "prompt", "test", "entry_point"

    Notes:
        "task_id" is the identifier string for the task.
        "prompt" is the prompt to be used for the task (function signature with docstrings).
        "test" is test-cases wrapped in a `check` function.
        "entry_point" is the name of the function.
    """
    # Check if open eval file exists in CACHE_DIR
    wild_code_bench_path = os.path.join(CACHE_DIR, "WildCodeBench.jsonl")
    make_cache(
        "https://github.com/bigcode-project/wild-code-bench-annotation/raw/main/data/wild-code-bench.jsonl.gz",
        wild_code_bench_path,
    )
    print(wild_code_bench_path)
    wild_code_bench = open(wild_code_bench_path, "r").read().split("\n")
    wild_code_bench = [json.loads(line) for line in wild_code_bench if line]

    return {task["task_id"]: task for task in wild_code_bench}

def get_wild_code_bench_hash() -> str:
    """Get the hash of WildCodeBench.
    Returns:
        str: The hash of WildCodeBench
    """
    wild_code_bench_path = os.path.join(CACHE_DIR, "WildCodeBench.jsonl")
    with open(wild_code_bench_path, "rb") as f:
        wild_code_bench = f.read()
    return hashlib.md5(wild_code_bench).hexdigest()