import hashlib
import json
import os
from typing import Dict

from bigcodebench.data.utils import (
    CACHE_DIR,
    completeness_check,
    get_dataset_metadata,
    make_cache,
    stream_jsonl,
)

BIGCODEBENCH_OVERRIDE_PATH = os.environ.get("BIGCODEBENCH_OVERRIDE_PATH", None)
BIGCODEBENCH_VERSION = "v0.1.0"

def _ready_bigcodebench_path(mini=False, noextreme=False, version="default") -> str:
    if BIGCODEBENCH_OVERRIDE_PATH:
        return BIGCODEBENCH_OVERRIDE_PATH

    version = BIGCODEBENCH_VERSION if version == "default" else version
    url, path = get_dataset_metadata(
        "BigCodeBench", BIGCODEBENCH_VERSION, mini, noextreme
    )
    make_cache(url, path)

    return path


def get_bigcodebench(
    err_incomplete=True, mini=False, noextreme=False, version="default"
    ) -> Dict[str, Dict]:
    """Get BigCodeBench from BigCode's github repo and return as a list of parsed dicts.

    Returns:
        List[Dict[str, str]]: List of dicts with keys "prompt", "test", "entry_point"

    Notes:
        "task_id" is the identifier string for the task.
        "prompt" is the prompt to be used for the task (function signature with docstrings).
        "test" is test-cases wrapped in a `check` function.
        "entry_point" is the name of the function.
    """
    # Check if open eval file exists in CACHE_DIR
    data_path = _ready_bigcodebench_path(
        mini=mini, noextreme=noextreme, version=version
    )
    data = {task["task_id"]: task for task in stream_jsonl(data_path)}
    if err_incomplete:
        completeness_check("BigCodeBench", data)
    return data

def get_bigcodebench_hash(mini=False, noextreme=False, version="default") -> str:
    """Get the hash of BigCodeBench.
    Returns:
        str: The hash of BigCodeBench
    """
    data_path = _ready_bigcodebench_path(mini, noextreme, version="default")
    with open(data_path, "rb") as f:
        data = f.read()
    return hashlib.md5(data).hexdigest()
