from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi

import json
import copy

BIGCODEBENCH_HF = "bigcode/bigcodebench"
BIGCODEBENCH_HARD_HF = "bigcode/bigcodebench-hard"
BIGCODEBENCH_VERSION = "v0.1.2"
BIGCODEBENCH_UPDATE = "bigcode/bcb_update"
BIGCODEBENCH_NEW_VERSION = "v0.1.3"

def map_ds(sample):
    if sample["task_id"] in ["BigCodeBench/1005"]:
        for k in sample.keys():
            sample[k] = sample[k].replace(
                "https://getsamplefiles.com/download/zip/sample-2.zip", "https://getsamplefiles.com/download/zip/sample-5.zip"
            ).replace(
                "sample_2", "sample_5"
            ).replace(
                "Sample 2", "Sample 5"
            )
    return sample
    
if __name__ == "__main__":
    api = HfApi()
    ds_dict = load_dataset(BIGCODEBENCH_HF)
    hard_ds_dict = load_dataset(BIGCODEBENCH_HARD_HF)
    ds = ds_dict[BIGCODEBENCH_VERSION]
    hard_ds = hard_ds_dict[BIGCODEBENCH_VERSION]
    function_id = [1005]
    
    new_ds = ds.map(map_ds)
    new_ds.to_json("BigCodeBench.jsonl")
    ds_dict[BIGCODEBENCH_NEW_VERSION] = new_ds
    ds_dict.push_to_hub(BIGCODEBENCH_HF)
    
    new_hard_ds = hard_ds.map(map_ds)
    new_hard_ds.to_json("BigCodeBench-Hard.jsonl")
    hard_ds_dict[BIGCODEBENCH_NEW_VERSION] = new_hard_ds
    hard_ds_dict.push_to_hub(BIGCODEBENCH_HARD_HF)
    
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
