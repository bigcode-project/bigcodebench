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
from datasets import load_dataset

BIGCODEBENCH_OVERRIDE_PATH = os.environ.get("BIGCODEBENCH_OVERRIDE_PATH", None)
BIGCODEBENCH_HF = "bigcode/bigcodebench"
BIGCODEBENCH_VERSION = "v0.1.0_hf"

def _ready_bigcodebench_path(mini=False, noextreme=False, version="default", offline=False) -> str:
    if BIGCODEBENCH_OVERRIDE_PATH:
        return BIGCODEBENCH_OVERRIDE_PATH

    version = BIGCODEBENCH_VERSION if version == "default" else version
    url, path = get_dataset_metadata(
        "BigCodeBench", BIGCODEBENCH_VERSION, mini, noextreme
    )
    
    try:
        dataset = load_dataset(BIGCODEBENCH_HF, split=BIGCODEBENCH_VERSION)
        if offline:
            with open("network-free-set.txt", "r") as f:
                included_ids = f.read()
            included_ids = included_ids.split("\n")
            dataset = dataset.filter(lambda instance: instance["task_id"] in included_ids)
        make_cache(url, dataset, path)
    except:
        if os.path.exists(path):
            os.remove(path)
        make_cache(url, None, path, gh=True)

    return path


def get_bigcodebench(
    err_incomplete=True, mini=False, noextreme=False, version="default", offline=False
    ) -> Dict[str, Dict]:
    """Get BigCodeBench from BigCode's github repo and return as a list of parsed dicts.

    Returns:
        List[Dict[str, str]]: List of dicts with keys "complete_prompt", "instruct_prompt", "canonical_solution", "test", "entry_point"

    Notes:
        "task_id" is the identifier string for the task.
        "complete_prompt" is the prompt to be used for BigCodeBench-Complete.
        "instruct_prompt" is the prompt to be used for BigCodeBench-Instruct.
        "canonical_solution" is the ground-truth implementation
        "test" is the `unittest.TestCase` class.
        "entry_point" is the name of the function.
    """
    # Check if open eval file exists in CACHE_DIR
    data_path = _ready_bigcodebench_path(
        mini=mini, noextreme=noextreme, version=version, offline=offline
    )
    data = {task["task_id"]: task for task in stream_jsonl(data_path)}
    if err_incomplete:
        completeness_check("BigCodeBench", data)
    return data

def get_bigcodebench_hash(mini=False, noextreme=False, version="default", offline=False) -> str:
    """Get the hash of BigCodeBench.
    Returns:
        str: The hash of BigCodeBench
    """
    data_path = _ready_bigcodebench_path(mini, noextreme, version, offline)
    with open(data_path, "rb") as f:
        data = f.read()
    return hashlib.md5(data).hexdigest()
