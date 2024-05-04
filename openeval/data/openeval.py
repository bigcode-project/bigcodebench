import hashlib
import json
import os
from typing import Dict

from openeval.data.utils import (
    CACHE_DIR,
    completeness_check,
    get_dataset_metadata,
    make_cache,
    stream_jsonl,
)

HUMANEVAL_OVERRIDE_PATH = os.environ.get("HUMANEVAL_OVERRIDE_PATH", None)


def get_open_eval() -> Dict[str, Dict]:
    """Get OpenEval from OpenAI's github repo and return as a list of parsed dicts.

    Returns:
        List[Dict[str, str]]: List of dicts with keys "prompt", "test", "entry_point"

    Notes:
        "task_id" is the identifier string for the task.
        "prompt" is the prompt to be used for the task (function signature with docstrings).
        "test" is test-cases wrapped in a `check` function.
        "entry_point" is the name of the function.
    """
    # Check if open eval file exists in CACHE_DIR
    open_eval_path = os.path.join(CACHE_DIR, "OpenEval.jsonl")
    # remove the cache
    # if os.path.exists(open_eval_path):
    #     os.remove(open_eval_path)
    make_cache(
        "https://github.com/bigcode-project/open-eval/raw/main/data/open-eval.jsonl.gz",
        open_eval_path,
    )

    open_eval = open(open_eval_path, "r").read().split("\n")
    open_eval = [json.loads(line) for line in open_eval if line]

    return {task["task_id"]: task for task in open_eval}

def get_open_eval_hash() -> str:
    """Get the hash of OpenEval.
    Returns:
        str: The hash of OpenEval
    """
    open_eval_path = os.path.join(CACHE_DIR, "OpenEval.jsonl")
    with open(open_eval_path, "rb") as f:
        open_eval = f.read()
    return hashlib.md5(open_eval).hexdigest()