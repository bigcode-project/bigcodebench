import json
import os
import shutil
import numpy as np
from numpy import mean
from glob import glob
from utils import *
from tqdm import tqdm
import pandas as pd
import itertools
import math
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from cuml.linear_model import LogisticRegression
import cupy as cp

def update_model_info(model_info):
    for model, info in model_info.items():
        if "https://huggingface.co/" in info["link"]:
            hf_model = info["link"].split("https://huggingface.co/")[-1]
            print(hf_model)
            try:
                tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
                
                if tokenizer.chat_template is None:
                    model_info[model]["direct_complete"] = True
                else:
                    model_info[model]["direct_complete"] = False
            except:
                model_info[model]["direct_complete"] = True
        else:
            model_info[model]["direct_complete"] = False
    
    return model_info


def get_results(tids):
    results = {}
    for model, info in model_info.items():
        results[info["name"]] = {
            "link": info["link"],
            "open-data": info["open-data"],
            "pass@1": {
                "complete": None,
                "instruct": None,
                "complete-cal": None,
                "instruct-cal": None,
            },
            "prompted": info["prompted"],
            "moe": info["moe"],
            "size": info["size"],
            "act_param": info["act_param"],
            # "direct_complete": info["direct_complete"],
        }
        
    for model, info in model_info.items():
        model = model.replace("/", "--")
        hf_model = ""
        files = glob(f"results/{model}--bigcodebench-*.json")
        assert files, f"No files found for results/{model}--bigcodebench-*.json"
        for file in files:
            try:
                _, suffix = os.path.basename(file).split("--bigcodebench-hard-")
                with open("results/"+model+"--bigcodebench-hard-"+suffix, "r") as f:
                    data = json.load(f)
            except:
                _, suffix = os.path.basename(file).split("--bigcodebench-")
                with open("results/"+model+"--bigcodebench-"+suffix, "r") as f:
                    data = json.load(f)
            status = []
            
            if len(data["eval"]) < len(tids):
                continue
            for key, value in data["eval"].items():
                if key not in tids:
                    continue
                if value[0]["status"] == "pass":
                    status.append(1)
                else:
                    status.append(0)
            if suffix.startswith("complete"):
                task = "complete"
            elif suffix.startswith("instruct"):
                task = "instruct"
            else:
                raise ValueError("Unknown task")

            mode = ""
            if "-sanitized-calibrate" in file:
                mode = "-cal"
            
            results[info["name"]][f"pass@1"][f"{task}{mode}"] = round(mean(status)*100,1)
            if not info["prompted"]:# or info["direct_complete"]:
                results[info["name"]][f"pass@1"][f"{task}-cal"] = round(mean(status)*100,1)
            
    for model, result in results.items():
        for task in ["complete"]:
            origin = result["pass@1"].pop(task)
            # assert origin, f"Missing original complete results for {model}"
            calibrate = result["pass@1"].pop(f"{task}-cal")
            if calibrate:
                # if calibrate - origin > 1:
                #     results[model]["lazy"] = True
                # else:
                #     results[model]["lazy"] = False
                results[model]["pass@1"][task] = calibrate
            else:
                # results[model]["lazy"] = False
                results[model]["pass@1"][task] = origin
        calibrate_instruct = result["pass@1"].pop(f"instruct-cal")
        result["pass@1"]["instruct"] = calibrate_instruct
    return results


def check_valid(results):
    for model, result in results.items():
        if result["prompted"] and model not in ["Granite-Code-3B-Instruct", "Granite-Code-8B-Instruct"]:
            assert result["pass@1"]["instruct"], model
        assert result["pass@1"]["complete"]


def split_gen():
    shutil.rmtree("sanitized_samples", ignore_errors=True)
    shutil.rmtree("sanitized_calibrated_samples", ignore_errors=True)
    os.makedirs("sanitized_samples/complete", exist_ok=True)
    os.makedirs("sanitized_samples/instruct", exist_ok=True)
    os.makedirs("sanitized_calibrated_samples/complete", exist_ok=True)
    os.makedirs("sanitized_calibrated_samples/instruct", exist_ok=True)
    for model, info in model_info.items():
        model = model.replace("/", "--")
        files = glob(f"results/{model}--bigcodebench-*.jsonl")
        if info["link"].startswith("https://huggingface.co/"):
            model = info["link"].split("https://huggingface.co/")[-1].replace("/", "--")
        
        for file in files:
            _, suffix = os.path.basename(file).split("--bigcodebench-")
            with open(file, "r") as f:
                data = f.readlines()
                
            if "-sanitized" in file:
                if "calibrated" in file:
                    if info["prompted"]:
                        if suffix.startswith("complete"):
                            with open(f"sanitized_calibrated_samples/complete/{model}--bigcodebench-{suffix}", "w") as f:
                                f.writelines(data)
                        else:
                            with open(f"sanitized_calibrated_samples/instruct/{model}--bigcodebench-{suffix}", "w") as f:
                                f.writelines(data)
                else:
                    if suffix.startswith("complete"):
                        with open(f"sanitized_samples/complete/{model}--bigcodebench-{suffix}", "w") as f:
                            f.writelines(data)
                    else:
                        with open(f"sanitized_samples/instruct/{model}--bigcodebench-{suffix}", "w") as f:
                            f.writelines(data)


def read_task_perf(tids, task="complete"):
    model_results = dict()
    result_files = []
    for model, info in model_info.items():
        if task == "instruct" and (not info["prompted"] or info["name"] in ["Granite-Code-3B-Instruct", "Granite-Code-8B-Instruct"]):
            continue

        task_perf = dict()
        model = model.replace("/", "--")
        try:
            try:
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
                    if info["prompted"]:# and not info["direct_complete"]:
                        files = glob(f"results/{model}--bigcodebench-{task}*-0-1-sanitized-calibrated_hard_eval_results.json")
                        if files:
                            file = files[0]
                        else:
                            file = glob(f"results/{model}--bigcodebench-{task}*-0-1-sanitized_hard_eval_results.json")[0]
                    else:
                        file = glob(f"results/{model}--bigcodebench-{task}*-0-1-sanitized_hard_eval_results.json")[0]
            except:
                try:
                    if info["prompted"]:# and not info["direct_complete"]:
                        files = glob(f"results/{model}--bigcodebench-hard-{task}*-0-1-sanitized-calibrated_hard_eval_results.json")
                        if files:
                            file = files[0]
                        else:
                            file = glob(f"results/{model}--bigcodebench-hard-{task}*-0-1-sanitized_hard_eval_results.json")[0]
                    else:
                        file = glob(f"results/{model}--bigcodebench-hard-{task}*-0-1-sanitized_hard_eval_results.json")[0]
                except:
                    if info["prompted"]:
                        files = glob(f"results/{model}--bigcodebench-hard-{task}*-0-1-sanitized-calibrated_eval_results.json")
                        if files:
                            file = files[0]
                        else:
                            file = glob(f"results/{model}--bigcodebench-hard-{task}*-0-1-sanitized_eval_results.json")[0]
                    else:
                        file = glob(f"results/{model}--bigcodebench-hard-{task}*-0-1-sanitized_eval_results.json")[0]
        except:
            continue
        
        result_files.append(file)
        with open(file, "r") as f:
            data = json.load(f)

        if len(data["eval"]) < len(tids):
            continue
        for task_id, perfs in data["eval"].items():
            if task_id in tids:
                status = 1 if perfs[0]["status"] == "pass" else 0
                task_perf[task_id] = status
        model_results[info["name"]] = task_perf
    return model_results, result_files


def get_winner_df(data_dict, tids, task, task_level=True, no_tie=True):
    winner_dict = {"task_id": [], "model_a": [], "model_b": [], "winner": []}
    if not task_level:
        file = f"{task}_winner_df.csv"
    else:
        file = f"{task}_winner_task_df.csv"
    
    if task_level:
        for task_id in tqdm(tids):
            # pair without repetition (a, b) and (b, a) are the same
            for model_a, model_b in itertools.combinations(data_dict.keys(), 2):
                solve_rate_a = data_dict[model_a][task_id]
                solve_rate_b = data_dict[model_b][task_id]
                
                if solve_rate_a > solve_rate_b:
                    winner_dict["winner"].append("model_a")
                elif solve_rate_a < solve_rate_b:
                    winner_dict["winner"].append("model_b")
                else:
                    if no_tie:
                        continue
                    winner_dict["winner"].append("tie")
                    
                winner_dict["task_id"].append(task_id)
                winner_dict["model_a"].append(model_a)
                winner_dict["model_b"].append(model_b)
    else:
        data_dict = {model: np.mean(list(task_perf.values())) for model, task_perf in data_dict.items()}
        for model_a, model_b in itertools.combinations(data_dict.keys(), 2):
            solve_rate_a = data_dict[model_a]
            solve_rate_b = data_dict[model_b]
            
            if solve_rate_a > solve_rate_b:
                winner_dict["winner"].append("model_a")
            elif solve_rate_a < solve_rate_b:
                winner_dict["winner"].append("model_b")
            else:
                if no_tie:
                    continue
                winner_dict["winner"].append("tie")
            winner_dict["task_id"].append(task)
            winner_dict["model_a"].append(model_a)
            winner_dict["model_b"].append(model_b)

    df = pd.DataFrame(winner_dict)
    df.to_csv(file, index=False)
    return df


def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True)))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def get_elo_mle(df, SCALE=400, BASE=10, INIT_RATING=1000):


    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)
    p = len(models.index)
    n = df.shape[0]

    X = cp.zeros([n, p])
    X[cp.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[cp.arange(n), models[df["model_b"]]] = -math.log(BASE)

    Y = cp.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    lr = LogisticRegression(fit_intercept=False)
    lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    return pd.Series(cp.asnumpy(elo_scores), index=models.index).sort_values(ascending=False)


def update_elo_rating(results, elo_dict):
    for model, info in model_info.items():
        if info["name"] not in elo_dict:
            results[info["name"]]["elo_mle"] = None
        else:
            results[info["name"]]["elo_mle"] = elo_dict[info["name"]]
    return results


def get_domain_perf(data_dict, task2domain):
    domain_perfs = {
        "Model": [],
        "Computation": [],
        "General": [],
        "Visualization": [],
        "System": [],
        "Time": [],
        "Network": [],
        "Cryptography": []
    }
    for model, task_perf in data_dict.items():
        model_domain = {"Computation": [], "General": [], "Visualization": [], "System": [], "Time": [], "Network": [], "Cryptography": []}
        for task_id, status in task_perf.items():
            domains = task2domain[task_id]
            for domain in domains:
                model_domain[domain].append(status)
        domain_perf = {domain: round(np.mean(perfs)*100, 1) for domain, perfs in model_domain.items()}
        domain_perfs["Model"].append(model)
        for domain in model_domain.keys():
            domain_perfs[domain].append(domain_perf[domain])
    return Dataset.from_dict(domain_perfs)


def get_solve_rate(data_dict, task="complete"):
    task_solve_count = dict()
    for model, task_perf in data_dict.items():
        for task_id, score in task_perf.items():
            if task_id not in task_solve_count:
                task_solve_count[task_id] = []
            task_solve_count[task_id].append(score)
    solve_rate = {task_id: round(np.mean(perfs) * 100, 1) for task_id, perfs in task_solve_count.items()}
    return Dataset.from_dict({"task_id": list(solve_rate.keys()), "solve_rate": list(solve_rate.values())})


def get_hf_ds(results):
    hf_dataset = {"model": [], "link": [], "moe": [], "size": [], "act_param": [], "type": [], #"lazy": [],# "direct_complete": [],
                  "complete": [], "instruct": [], "elo_mle": []}

    for model, result in results.items():
        hf_dataset["model"].append(model)
        hf_dataset["link"].append(result["link"])
        hf_dataset["moe"].append(result["moe"])
        hf_dataset["size"].append(result["size"])
        hf_dataset["act_param"].append(result["act_param"])
        hf_dataset["type"].append("🔶" if result["prompted"] else "🟢")
        # hf_dataset["lazy"].append(result["lazy"])
        hf_dataset["complete"].append(result["pass@1"]["complete"])
        hf_dataset["instruct"].append(result["pass@1"]["instruct"])
        # hf_dataset["direct_complete"].append(result["direct_complete"])
        hf_dataset["elo_mle"].append(result["elo_mle"])

    return Dataset.from_dict(hf_dataset)

def get_bootstrap_scores(df):
    bars = pd.DataFrame(dict(
        lower = df.quantile(.025),
        rating = df.quantile(.5),
        upper = df.quantile(.975))).reset_index(names="model").sort_values("rating", ascending=False)
    
    bars['error_y'] = bars['upper'] - bars["rating"]
    bars['error_y_minus'] = bars['rating'] - bars["lower"]
    bars['rating_rounded'] = np.round(bars['rating'], 2)
    return Dataset.from_pandas(bars)


def push_ds(ds, path, local=False):
    if local:
        ds.save_to_disk(path)
    else:
        ds.push_to_hub(path)


def get_perf_df(data_dict):
    perfs = {"Model": []}
    for task_id in data_dict[list(data_dict.keys())[0]]:
        perfs[task_id] = []
    for model, task_perf in data_dict.items():
        perfs["Model"].append(model)
        for task_id, status in task_perf.items():
            perfs[task_id].append(status)
    return pd.DataFrame(perfs)

    
if __name__ == "__main__":
    
    bcb_orig = load_dataset("bigcode/bigcodebench", split="v0.1.1")
    bcb_hard = load_dataset("bigcode/bigcodebench-hard", split="v0.1.1")
    bcb_config = {
        "": bcb_orig,
        "-hard": bcb_hard,
    }
    for suffix, bcb in bcb_config.items():
        results = get_results(bcb["task_id"])
        files = []
        complete_data, complete_files = read_task_perf(bcb["task_id"], "complete")
        instruct_data, instruct_files = read_task_perf(bcb["task_id"], "instruct")
        complete_df = get_perf_df(complete_data)
        instruct_df = get_perf_df(instruct_data)
        
        push_ds(DatasetDict({"complete": Dataset.from_pandas(complete_df), "instruct": Dataset.from_pandas(instruct_df)}), f"bigcode/bigcodebench{suffix}-perf")

        with open("task2domain.json", "r") as f:
            task2domain = json.load(f)
        domain_complete = get_domain_perf(complete_data, task2domain)
        domain_instruct = get_domain_perf(instruct_data, task2domain)
        DatasetDict({"complete": domain_complete, "instruct": domain_instruct}).push_to_hub(f"bigcode/bigcodebench{suffix}-domain")

        files.extend(complete_files)
        files.extend(instruct_files)
        shutil.rmtree("eval_results", ignore_errors=True)
        os.makedirs("eval_results", exist_ok=True)
        for file in files:
            shutil.copy(file, "eval_results")
        
        complete_solve_rate = get_solve_rate(complete_data, task="complete")
        instruct_solve_rate = get_solve_rate(instruct_data, task="instruct")
        solve_rate_ds = DatasetDict({"complete": complete_solve_rate, "instruct": instruct_solve_rate})
        push_ds(solve_rate_ds, f"bigcode/bigcodebench{suffix}-solve-rate")
        
        elo_config = {
            "task_no_tie": (True, True),
            "benchmark_tie": (False, False),
        }
        elo_ds = dict()
        for config, (task_level, no_tie) in elo_config.items():
            filter_complete_data = {model: task_perf for model, task_perf in complete_data.items() if model in instruct_data}
            complete_battles = get_winner_df(filter_complete_data, bcb["task_id"], "complete", task_level=task_level, no_tie=no_tie)
            instruct_battles = get_winner_df(instruct_data, bcb["task_id"], "instruct", task_level=task_level, no_tie=no_tie)
            battles = pd.concat([complete_battles, instruct_battles])
            elo_mle_bootstrap = get_bootstrap_result(battles, get_elo_mle, 500)
            bootstrap_lu_median = elo_mle_bootstrap.median().reset_index().set_axis(["model", "Elo rating"], axis=1)
            bootstrap_lu_median["Elo rating"] = (bootstrap_lu_median["Elo rating"] + 0.5).astype(int)
            bootstrap_lu_median_dict = bootstrap_lu_median.set_index("model")["Elo rating"].to_dict()
            if config == "task_no_tie":
                task_elo = bootstrap_lu_median_dict
            elo = get_bootstrap_scores(elo_mle_bootstrap)
            elo_ds[config] = elo
        push_ds(DatasetDict(elo_ds), f"bigcode/bigcodebench{suffix}-elo")

        results = update_elo_rating(results, task_elo)
        with open(f"results{suffix}.json", "w") as f:
            json.dump(results, f, indent=4)
        ds = get_hf_ds(results)
        push_ds(ds, f"bigcode/bigcodebench{suffix}-results")