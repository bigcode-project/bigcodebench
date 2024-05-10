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


def get_wildcodebench() -> Dict[str, Dict]:
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
    wildcodebench_path = os.path.join(CACHE_DIR, "WildCodeBench.jsonl")
    make_cache(
        "https://github.com/bigcode-project/wild-code-bench-annotation/raw/main/data/wild-code-bench.jsonl.gz",
        wildcodebench_path,
    )
    wildcodebench = open(wildcodebench_path, "r").read().split("\n")
    wildcodebench = [json.loads(line) for line in wildcodebench if line]

    return {task["task_id"]: task for task in wildcodebench}

def get_wildcodebench_hash() -> str:
    """Get the hash of WildCodeBench.
    Returns:
        str: The hash of WildCodeBench
    """
    wildcodebench_path = os.path.join(CACHE_DIR, "WildCodeBench.jsonl")
    with open(wildcodebench_path, "rb") as f:
        wildcodebench = f.read()
    return hashlib.md5(wildcodebench).hexdigest()