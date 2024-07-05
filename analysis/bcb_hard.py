import pickle
import json
import numpy as np
from tqdm import tqdm
from ast import literal_eval
from glob import glob
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset, Features, Value, Sequence, DatasetDict

from utils import *

def embed_sentences(data, col_name, id_name, model, save_path, push_to_hub=False):
    pool = model.start_multi_process_pool()
    embeddings = model.encode_multi_process(data[col_name], pool=pool)
    qids = data[id_name]
    features = Features({id_name: Value(dtype='string'), 'embeddings': Sequence(Value('float32'))})
    embed_dict = {
        id_name: qids,
        "embeddings": embeddings
    }
    embed_ds = Dataset.from_dict(embed_dict, features=features)
    if push_to_hub:
        embed_ds.push_to_hub(f"bigcode/{save_path}")
    else:
        embed_ds.save_to_disk(save_path)
    return embed_ds


def get_top_docs(query_embs, doc_emb, docs):
    scores = np.dot(query_embs, doc_emb.T)
    top_doc_indices = np.argmax(scores, axis=1)
    top_scores = scores[np.arange(len(scores)), top_doc_indices]
    results = [(i, docs[doc_idx], score) for i, (doc_idx, score) in tqdm(enumerate(zip(top_doc_indices, top_scores)))]
    
    return results


def filter_top_k_percent(results, k_percent):
    all_scores = [score for _, score in results]
    threshold = np.percentile(all_scores, 100 - k_percent)
    filtered_results = [(i, doc, score) for i, doc, score in results if score > threshold]
    return filtered_results


def filter_top_threshold(results, threshold):
    filtered_results = [(i, doc, score) for i, doc, score in results if score > threshold]
    return filtered_results


def read_task_perf(top_tid, task="complete"):
    model_results = dict()
    result_files = []
    for model, info in model_info.items():
        if task == "instruct" and (not info["prompted"] or info["name"] in ["Granite-Code-3B-Instruct", "Granite-Code-8B-Instruct"]):
            continue
        task_perf = {f"BigCodeBench/{task_id}": 0 for task_id in range(1140)}
        model = model.replace("/", "--")
        if info["link"].startswith("https://huggingface.co/"):
            model = info["link"].split("https://huggingface.co/")[-1].replace("/", "--")
        try:
            if info["prompted"]:
                files = glob(f"results/{model}--bigcodebench-{task}*-0-1-sanitized-calibrated_eval_results.json")
                if files:
                    file = files[0]
                else:
                    file = glob(f"results/{model}--bigcodebench-{task}*-0-1-sanitized_eval_results.json")[0]
            else:
                file = glob(f"results/{model}--bigcodebench-{task}*-0-1-sanitized_eval_results.json")[0]
        except:
            continue
        with open(file, "r") as f:
            data = json.load(f)
        for task_id, perfs in data["eval"].items():
            status = 1 if perfs[0]["status"] == "pass" else 0
            task_perf[task_id] = status
        model_results[info["name"]] = np.mean([status for tid, status in task_perf.items() if tid in top_tid])
    return sorted(model_results.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":    
    bcb = load_dataset("bigcode/bigcodebench", trust_remote_code=True, split="v0.1.0_hf")
    se = load_dataset("bigcode/stack-exchange-preferences-20230914-clean-anonymization", trust_remote_code=True, split="train")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    se_embed = embed_sentences(se, "question", "qid", model, "stack-exchange-embeddings-20230914", push_to_hub=True)
    bcb_embed = embed_sentences(bcb, "complete_prompt", "task_id", model, "bigcodebench-doc-embeddings", push_to_hub=True)

    solve_rate = load_dataset("bigcode/bigcodebench-solve-rate", trust_remote_code=True, split="complete")

    query_embs = np.array(se_embed["embeddings"])
    doc_emb = np.array(bcb_embed["embeddings"])
    docs = bcb_embed["task_id"]
    retrieval_results = get_top_docs(query_embs, doc_emb, docs)

    Dataset.from_dict({"qid": [i for i, _, _ in retrieval_results], "tid": [doc for _, doc, _ in retrieval_results], "score": [score for _, _, score in retrieval_results]}).push_to_hub("bigcode/se_bcb_results")

    retrieval_ds = load_dataset("bigcode/se_bcb_results", trust_remote_code=True, split="train")
    retrieval_ds = load_dataset("bigcode/se_bcb_instruct_results", trust_remote_code=True, split="train")

    top_results = dict()
    for sample in tqdm(retrieval_ds):
        i, doc, score = sample["qid"], sample["tid"], sample["score"]
        if score > 0.7:
            if doc not in top_results:
                top_results[doc] = (i, doc, score)
            else:
                if score > top_results[doc][2]:
                    top_results[doc] = (i, doc, score)

    top_id = {task_id: (qid, score) for qid, task_id, score in top_results.values()}

    lib_filter = {sample["task_id"] for sample in bcb if len(literal_eval(sample["libs"])) > 2}
    length_filter = {sample["task_id"] for sample in bcb if len(sample["canonical_solution"]) > 426}
    rate_filter = {task["task_id"]: task["solve_rate"] for task in solve_rate if task["solve_rate"] < 50}
            
    top_tid = top_id.keys() & length_filter & rate_filter.keys() & lib_filter
    # hard_results = read_task_perf(top_tid)

    filtered_bcb = bcb.filter(lambda x: x["task_id"] in top_tid)
    hard_bcb_tid = hard_bcb["task_id"]
    se_qid = [top_id[_id][0] for _id in hard_bcb_tid]
    se_q = se.select(se_qid)
    se_scores = [top_id[_id][1] for _id in hard_bcb_tid]
    
    hard_bcb_dict = {
        "task_id": [f"BigCodeBenchHard/{i}" for i in range(len(hard_bcb))],
        "complete_prompt": hard_bcb["complete_prompt"],
        "instruct_prompt": hard_bcb["instruct_prompt"],
        "canonical_solution": hard_bcb["canonical_solution"],
        "code_prompt": hard_bcb["code_prompt"],
        "test": hard_bcb["test"],
        "entry_point": hard_bcb["entry_point"],
        "doc_struct": hard_bcb["doc_struct"],
        "libs": hard_bcb["libs"],
        "q_idx": se_qid,
        "question": se_q["question"],
        "score": se_scores,
        "_id": hard_bcb_tid
    }
    
    hard_bcb = Dataset.from_dict(hard_bcb_dict)
    DatasetDict({"v0.1.0_hf": hard_bcb}).push_to_hub("bigcode/bigcodebench-hard")