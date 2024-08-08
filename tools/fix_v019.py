from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi

import json
import copy

BIGCODEBENCH_HF = "bigcode/bigcodebench"
BIGCODEBENCH_VERSION = "v0.1.0_hf"
BIGCODEBENCH_UPDATE = "bigcode/bcb_update"
BIGCODEBENCH_NEW_VERSION = "v0.1.1"

def map_ds(sample):
    if sample["task_id"] in ["BigCodeBench/1005", "BigCodeBench/1006"]:
        sample["test"] = sample["test"].replace(
            "https://getsamplefiles.com/download/zip/",
            "https://github.com/bigcode-project/bigcodebench-annotation/releases/download/v0.1.0_hf/"
        )
    
    if sample["task_id"] in ["BigCodeBench/760"]:
        for k in sample.keys():
            if "prompt" in k:
                sample[k] = sample[k].replace(
                    "from datetime import datetime",
                    "import datetime"
                )
    
    return sample
    
if __name__ == "__main__":
    api = HfApi()
    ds_dict = load_dataset(BIGCODEBENCH_HF)
    ds = ds_dict[BIGCODEBENCH_VERSION]
    function_id = [760, 1005, 1006]
    
    new_ds = ds.map(map_ds)
    new_ds.to_json("new_ds.jsonl")
    ds_dict[BIGCODEBENCH_NEW_VERSION] = new_ds
    
    for i in function_id:
        old_sample = ds.select([i])
        new_sample = new_ds.select([i])
        old_sample.to_json("old.jsonl")
        new_sample.to_json("new.jsonl")
        api.upload_file(
            path_or_fileobj="old.jsonl",
            path_in_repo=f"{i}/old.jsonl",
            repo_id=BIGCODEBENCH_UPDATE,
            # repo_type="dataset"
        )
        api.upload_file(
            path_or_fileobj="new.jsonl",
            path_in_repo=f"{i}/new.jsonl",
            repo_id=BIGCODEBENCH_UPDATE,
            # repo_type="dataset"
        )
    
