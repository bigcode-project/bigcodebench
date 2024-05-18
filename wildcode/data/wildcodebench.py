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

WILDCODEBENCH_OVERRIDE_PATH = os.environ.get("WILDCODEBENCH_OVERRIDE_PATH", None)
WILDCODEBENCH_VERSION = "v0.1.1"

def _ready_wildcodebench_path(mini=False, noextreme=False, version="default") -> str:
    if WILDCODEBENCH_OVERRIDE_PATH:
        return WILDCODEBENCH_OVERRIDE_PATH

    version = WILDCODEBENCH_VERSION if version == "default" else version
    url, path = get_dataset_metadata(
        "WildCodeBench", WILDCODEBENCH_VERSION, mini, noextreme
    )
    make_cache(url, path)

    return path


def get_wildcodebench(
    err_incomplete=True, mini=False, noextreme=False, version="default"
    ) -> Dict[str, Dict]:
    """Get WildCodeBench from BigCode's github repo and return as a list of parsed dicts.

    Returns:
        List[Dict[str, str]]: List of dicts with keys "prompt", "test", "entry_point"

    Notes:
        "task_id" is the identifier string for the task.
        "prompt" is the prompt to be used for the task (function signature with docstrings).
        "test" is test-cases wrapped in a `check` function.
        "entry_point" is the name of the function.
    """
    # Check if open eval file exists in CACHE_DIR
    data_path = _ready_wildcodebench_path(
        mini=mini, noextreme=noextreme, version=version
    )
    data = {task["task_id"]: task for task in stream_jsonl(data_path)}
    if err_incomplete:
        completeness_check("WildCodeBench", data)
    return data

def get_wildcodebench_hash(mini=False, noextreme=False, version="default") -> str:
    """Get the hash of WildCodeBench.
    Returns:
        str: The hash of WildCodeBench
    """
    data_path = _ready_wildcodebench_path(mini, noextreme, version="default")
    with open(data_path, "rb") as f:
        data = f.read()
    return hashlib.md5(data).hexdigest()
