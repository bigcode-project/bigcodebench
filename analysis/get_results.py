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
from datasets import Dataset


def get_results():
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
            "size": info["size"],
        }
        
    for model, info in model_info.items():
        model = model.replace("/", "--")
        if "https://huggingface.co/" in info["link"]:
            model = info["link"].split("https://huggingface.co/")[-1].replace("/", "--")
        files = glob(f"results/{model}--bigcodebench-*.json")
        assert files, f"No files found for results/{model}--bigcodebench-*.json"
        for file in files:
            # print(file)
            _, suffix = os.path.basename(file).split("--bigcodebench-")
            status = []
            with open("results/"+model+"--bigcodebench-"+suffix, "r") as f:
                data = json.load(f)
            if len(data["eval"]) != 1140:
                continue
            for key, value in data["eval"].items():
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
            if not info["prompted"]:
                results[info["name"]][f"pass@1"][f"{task}-cal"] = round(mean(status)*100,1)
        
    for model, result in results.items():
        for task in ["complete"]:
            origin = result["pass@1"].pop(task)
            assert origin, f"Missing original complete results for {model}"
            calibrate = result["pass@1"].pop(f"{task}-cal")
            assert calibrate, f"Missing calibrated complete results for {model}"
            if calibrate - origin > 1:
                results[model]["lazy"] = True
            else:
                results[model]["lazy"] = False
            results[model]["pass@1"][task] = calibrate
        calibrate_instruct = result["pass@1"].pop(f"instruct-cal")
        result["pass@1"]["instruct"] = calibrate_instruct
    return results

           
def compute_diff(results):
    diffs = []
    for model, info in model_info.items():
        if not info["prompted"]:
            continue
        diff = results[info["name"]]["pass@1"]["complete"] - results[info["name"]]["pass@1"]["complete-cal"]
        diffs.append(diff)
    print("Mean diff:", mean(diffs))


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
        files = glob(f"clean_results/{model}--bigcodebench-*.jsonl")
        if "https://huggingface.co/" in info["link"]:
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


def read_task_perf(task="complete"):
    model_results = dict()
    for model, info in model_info.items():
        if task == "instruct" and (not info["prompted"] or info["name"] in ["Granite-Code-3B-Instruct", "Granite-Code-8B-Instruct"]):
            continue

        task_perf = {f"BigCodeBench/{task_id}": 0 for task_id in range(1140)}
        model = model.replace("/", "--")
        if "https://huggingface.co/" in info["link"]:
            model = info["link"].split("https://huggingface.co/")[-1].replace("/", "--")
        try:
            if info["prompted"]:
                file = glob(f"results/{model}--bigcodebench-{task}*-0-1-sanitized-calibrated_eval_results.json")[0]
            else:
                file = glob(f"results/{model}--bigcodebench-{task}*-0-1-sanitized_eval_results.json")[0]
        except:
            continue
            raise ValueError(f"Missing results/{model}--bigcodebench-{task}*-0-1-sanitized_eval_results.json")
        
        with open(file, "r") as f:
            data = json.load(f)
        for task_id, perfs in data["eval"].items():
            status = 1 if perfs[0]["status"] == "pass" else 0
            task_perf[task_id] = status
        model_results[info["name"]] = task_perf
    return model_results


def get_winner_df(data_dict, task, task_level=True, no_tie=True):
    winner_dict = {"task_id": [], "model_a": [], "model_b": [], "winner": []}
    if not task_level:
        file = f"{task}_winner_df.csv"
    else:
        file = f"{task}_winner_task_df.csv"
    
    if task_level:
        for task_id in tqdm(range(1140)):
            task_id = f"BigCodeBench/{task_id}"
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
    from sklearn.linear_model import LogisticRegression
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    lr = LogisticRegression(fit_intercept=False)
    lr.fit(X,Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    return pd.Series(elo_scores, index = models.index).sort_values(ascending=False)


def update_elo_rating(results, elo_dict):
    for model, info in model_info.items():
        if info["name"] not in elo_dict:
            continue
        results[info["name"]]["elo_mle"] = elo_dict[info["name"]]
    return results


def get_solve_rate(data_dict, task="complete"):
    task_solve_count = {f"BigCodeBench/{task_id}": [] for task_id in range(1140)}
    for model, task_perf in data_dict.items():
        for task_id in range(1140):
            task_solve_count[f"BigCodeBench/{task_id}"].append(task_perf[f"BigCodeBench/{task_id}"])
    solve_rate = {task_id: round(np.mean(perfs) * 100, 1) for task_id, perfs in task_solve_count.items()}
    with open(f"{task}_solve_rate.txt", "w") as f:
        f.write(f"Number of unsolved tasks: {sum([1 for task_id, solve_rate in solve_rate.items() if solve_rate == 0])}\n")
        f.write(f"Number of fully solved tasks: {sum([1 for task_id, solve_rate in solve_rate.items() if solve_rate == 100])}\n")
    return Dataset.from_dict({"task_id": list(solve_rate.keys()), "solve_rate": list(solve_rate.values())})


def get_hf_ds(results):
    hf_dataset = {"model": [], "link": [], "size": [], "type": [], "lazy": [],
                  "complete": [], "instruct": [], "elo_mle": []}

    for model, result in results.items():
        hf_dataset["model"].append(model)
        hf_dataset["link"].append(result["link"])
        hf_dataset["size"].append(result["size"])
        hf_dataset["type"].append("🔶" if result["prompted"] else "🟢")
        hf_dataset["lazy"].append(result["lazy"])
        hf_dataset["complete"].append(result["pass@1"]["complete"])
        hf_dataset["instruct"].append(result["pass@1"]["instruct"])
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


if __name__ == "__main__":
    results = get_results()
    complete_data = read_task_perf("complete")
    instruct_data = read_task_perf("instruct")
    complete_solve_rate = get_solve_rate(complete_data, task="complete")
    instruct_solve_rate = get_solve_rate(instruct_data, task="instruct")
    push_ds(complete_solve_rate, "bigcode/bigcodebench-complete-solve-rate")
    push_ds(instruct_solve_rate, "bigcode/bigcodebench-instruct-solve-rate")
    
    battles = get_winner_df(complete_data, "complete")
    elo_mle_bootstrap = get_bootstrap_result(battles, get_elo_mle, 500)
    bootstrap_lu_median = elo_mle_bootstrap.median().reset_index().set_axis(["model", "Elo rating"], axis=1)
    bootstrap_lu_median["Elo rating"] = (bootstrap_lu_median["Elo rating"] + 0.5).astype(int)
    bootstrap_lu_median_dict = bootstrap_lu_median.set_index("model")["Elo rating"].to_dict()
    elo = get_bootstrap_scores(elo_mle_bootstrap)
    push_ds(elo, "bigcode/bigcodebench-elo")
    
    results = update_elo_rating(results, bootstrap_lu_median_dict)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    ds = get_hf_ds(results)
    push_ds(ds, "bigcode/bigcodebench-results")