# BigCodeBench
<center>
<img src="https://github.com/bigcode-bench/bigcode-bench.github.io/blob/main/asset/bigcodebench_banner.svg?raw=true" alt="BigCodeBench">
</center>

<p align="center">
    <a href="https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard"><img src="https://img.shields.io/badge/🤗&nbsp&nbsp%F0%9F%8F%86-leaderboard-%23ff8811"></a>
    <a href="https://huggingface.co/collections/bigcode/bigcodebench-666ed21a5039c618e608ab06"><img src="https://img.shields.io/badge/🤗-collection-pink"></a>
    <a href="https://bigcode-bench.github.io/"><img src="https://img.shields.io/badge/%F0%9F%8F%86-website-8A2BE2"></a>
    <a href="https://arxiv.org/abs/2406.15877"><img src="https://img.shields.io/badge/arXiv-2406.15877-b31b1b.svg"></a>
    <a href="https://pypi.org/project/bigcodebench/"><img src="https://img.shields.io/pypi/v/bigcodebench?color=g"></a>
    <a href="https://pepy.tech/project/bigcodebench"><img src="https://static.pepy.tech/badge/bigcodebench"></a>
    <a href="https://github.com/bigcodebench/bigcodebench/blob/master/LICENSE"><img src="https://img.shields.io/pypi/l/bigcodebench"></a>
    <a href="https://hub.docker.com/r/bigcodebench/bigcodebench-evaluate" title="Docker-Eval"><img src="https://img.shields.io/docker/image-size/bigcodebench/bigcodebench-evaluate"></a>
    <a href="https://hub.docker.com/r/bigcodebench/bigcodebench-generate" title="Docker-Gen"><img src="https://img.shields.io/docker/image-size/bigcodebench/bigcodebench-generate"></a>
</p>

<p align="center">
    <a href="#-about">🌸About</a> •
    <a href="#-quick-start">🔥Quick Start</a> •
    <a href="#-failure-inspection">🔍Failure Inspection</a> •
    <a href="#-full-script">🚀Full Script</a> •
    <a href="#-result-analysis">📊Result Analysis</a> •
    <a href="#-llm-generated-code">💻LLM-generated Code</a> •
    <a href="#-known-issues">🐞Known Issues</a> •
    <a href="#-citation">📜Citation</a> •
    <a href="#-acknowledgement">🙏Acknowledgement</a>
</p>

## 🌸 About

### BigCodeBench

BigCodeBench is an **_easy-to-use_** benchmark for code generation with **_practical_** and **_challenging_** programming tasks. It aims to evaluate the true programming capabilities of large language models (LLMs) in a more realistic setting. The benchmark is designed for HumanEval-like function-level code generation tasks, but with much more complex instructions and diverse function calls.
To facilitate the evaluation of LLMs on BigCodeBench, we provide this Python package `bigcodebench` that includes the dataset, generation scripts, and evaluation scripts. The package is built on top of the [EvalPlus](https://github.com/evalplus/evalplus) framework, which is a flexible and extensible evaluation framework for code generation tasks.

### Why BigCodeBench?

BigCodeBench focuses on the evaluation of LLM4Code with *diverse function calls* and *complex instruction*, with:

* ✨ **Precise evaluation & ranking**: See [our leaderboard](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard) for latest LLM rankings before & after rigorous evaluation.
* ✨ **Pre-generated samples**: BigCodeBench accelerates code intelligence research by open-sourcing [LLM-generated samples](#-LLM-generated-code) for various models -- no need to re-run the expensive benchmarks!

### Main Differences from EvalPlus

We inherit the design of the EvalPlus framework, which is a flexible and extensible evaluation framework for code generation tasks. However, BigCodeBench has the following differences:
* Execution Environment: The execution environment in BigCodeBench is less bounded than EvalPlus to support tasks with diverse library dependencies.
* Test Evaluation: BigCodeBench relies on `unittest` for evaluating the generated code, which is more suitable for the test harness in BigCodeBench.

## 🔥 Quick Start

> [!Tip]
>
> BigCodeBench ❤️ [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness)!
> BigCodeBench will be integrated to bigcode-evaluation-harness, and you can also run it there!

To get started, please first set up the environment:

```bash
# Install to use bigcodebench.evaluate
pip install bigcodebench --upgrade
# If you want to use the evaluate locally, you need to install the requirements
pip install -I -r https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt

# Install to use bigcodebench.generate
# You are strongly recommended to install the generate dependencies in a separate environment
pip install bigcodebench[generate] --upgrade
```

<details><summary>⏬ Install nightly version <i>:: click to expand ::</i></summary>
<div>

```bash
# Install to use bigcodebench.evaluate
pip install "git+https://github.com/bigcode-project/bigcodebench.git" --upgrade
```

</div>
</details>

<details><summary>⏬ Using BigCodeBench as a local repo? <i>:: click to expand ::</i></summary>
<div>

```bash
git clone https://github.com/bigcode-project/bigcodebench.git
cd bigcodebench
export PYTHONPATH=$PYTHONPATH:$(pwd)
# Install to use bigcodebench.evaluate
pip install -e .
# Install to use bigcodebench.generate
pip install -e .[generate]
```

</div>
</details>

### Code Generation

You are suggested to use `flash-attn` for generating code samples.
```bash
pip install -U flash-attn
```

To generate code samples from a model, you can use the following command:
>
```bash
# when greedy, there is no need for temperature and n_samples
bigcodebench.generate \
    --model [model_name] \
    --split [complete|instruct] \
    --subset [full|hard] \
    [--greedy] \
    --bs [bs] \
    --temperature [temp] \
    --n_samples [n_samples] \
    --resume \
    --backend [vllm|hf|openai|mistral|anthropic|google] \
    --tp [gpu_number] \
    [--trust_remote_code] \
    [--base_url [base_url]]
```
>
The generated code samples will be stored in a file named `[model_name]--bigcodebench-[instruct|complete]--[backend]-[temp]-[n_samples].jsonl`. Alternatively, you can use the following command to utilize our pre-built docker images for generating code samples:
>
```bash
# If you are using GPUs
docker run --gpus '"device=$CUDA_VISIBLE_DEVICES"' -v $(pwd):/app -t bigcodebench/bigcodebench-generate:latest \
    --model [model_name] \ 
    --split [complete|instruct] \
    --subset [full|hard] \
    [--greedy] \
    --bs [bs] \   
    --temperature [temp] \
    --n_samples [n_samples] \
    --resume \
    --backend [vllm|hf|openai|mistral|anthropic|google] \
    --tp [gpu_number]

# ...Or if you are using CPUs
docker run -v $(pwd):/app -t bigcodebench/bigcodebench-generate:latest \
    --model [model_name] \ 
    --split [complete|instruct] \
    --subset [full|hard] \
    [--greedy] \
    --bs [bs] \   
    --temperature [temp] \
    --n_samples [n_samples] \
    --resume \
    --backend [vllm|hf|openai|mistral|anthropic|google]
```
>
```bash
# If you wish to use gated or private HuggingFace models and datasets
docker run -e HUGGING_FACE_HUB_TOKEN=$token -v $(pwd):/app -t bigcodebench/bigcodebench-generate:latest # omit other arguments4

# Similarly, to use other backends that require authentication
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -v $(pwd):/app -t bigcodebench/bigcodebench-generate:latest # omit other arguments
docker run -e GOOGLE_API_KEY=$OPENAI_API_KEY -v $(pwd):/app -t bigcodebench/bigcodebench-generate:latest # omit other arguments
docker run -e ANTHROPIC_KEY=$ANTHROPIC_KEY -v $(pwd):/app -t bigcodebench/bigcodebench-generate:latest # omit other arguments
```
>
Following which, you can run the built container as shown in above.
>
<details><summary>🤔 Structure of `problem`? <i>:: click to expand ::</i></summary>
<div>

* `task_id` is the identifier string for the task
* `entry_point` is the name of the function
* `complete_prompt` is the prompt for BigCodeBench-Complete
* `instruct_prompt` is the prompt for BigCodeBench-Instruct
+ `canonical_solution` is the ground-truth implementation
+ `test` is the `unittest.TestCase` class

</div>
</details>

> [!Note]
>
> **Expected Schema of `[model_name]--bigcodebench-[task]--[backend]-[temp]-[n_samples].jsonl`**
>
> 1. `task_id`: Task ID, which are the keys of `get_bigcodebench()`
> 2. `solution` (optional): Self-contained solution (usually including the prompt)
>    * Example: `{"task_id": "BigCodeBench/?", "solution": "def f():\n    return 1"}`

### Code Post-processing

LLM-generated text may not be compilable code for including natural language lines or incomplete extra code.
We provide a tool namely `bigcodebench.sanitize` to clean up the code:

```bash
# 💡 If you want to get the calibrated results:
bigcodebench.sanitize --samples samples.jsonl --calibrate
# Sanitized code will be produced to `samples-sanitized-calibrated.jsonl`

# 💡 If you want to get the original results:
bigcodebench.sanitize --samples samples.jsonl
# Sanitized code will be produced to `samples-sanitized.jsonl`

# 💡 If you are storing codes in directories:
bigcodebench.sanitize --samples /path/to/vicuna-[??]b_temp_[??]
# Sanitized code will be produced to `/path/to/vicuna-[??]b_temp_[??]-sanitized`
```

If you want to use the pre-built docker images for post-processing, you can use the following command:

```bash
# Change the entrypoint to bigcodebench.sanitize in any pre-built docker image, like bigcodebench/bigcodebench-evaluate:latest
docker run -it --entrypoint bigcodebench.sanitize -v $(pwd):/app bigcodebench/bigcodebench-evaluate:latest --samples samples.jsonl
```

<details><summary>🔎 Checking the compatibility of post-processed code<i>:: click to expand ::</i></summary>
<div>

To double-check the post-processing results, you can use `bigcodebench.syncheck` to check the code validity before and after sanitization, which will print erroneous code snippets and why they are wrong:

```bash
# 💡 If you are storing codes in jsonl:
bigcodebench.syncheck --samples samples.jsonl

# 💡 If you are storing codes in directories:
bigcodebench.syncheck --samples /path/to/vicuna-[??]b_temp_[??]

# 💡 Or change the entrypoint to bigcodebench.syncheck in any pre-built docker image, like 
docker run -it --entrypoint bigcodebench.syncheck -v $(pwd):/app bigcodebench/bigcodebench-evaluate:latest --samples samples.jsonl
```

</div>
</details>


### Code Evaluation

You are strongly recommended to use a sandbox such as [docker](https://docs.docker.com/get-docker/):

```bash
# Mount the current directory to the container
# If you want to change the RAM address space limit (in MB, 128 GB by default): `--max-as-limit XXX`
# If you want to change the RAM data segment limit (in MB, 4 GB by default): `--max-data-limit`
# If you want to change the RAM stack limit (in MB, 4 MB by default): `--max-stack-limit`
docker run -v $(pwd):/app bigcodebench/bigcodebench-evaluate:latest --split [complete|instruct] --subset [full|hard] --samples samples-sanitized-calibrated.jsonl

# If you only want to check the ground truths
docker run -v $(pwd):/app bigcodebench/bigcodebench-evaluate:latest --split [complete|instruct] --subset [full|hard] --samples samples-sanitized-calibrated.jsonl --check-gt-only
```

...Or if you want to try it locally regardless of the risks ⚠️:

First, install the dependencies for BigCodeBench:

```bash
pip install -r https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt
```

Then, run the evaluation:

```bash
# ...Or locally ⚠️
bigcodebench.evaluate --split [complete|instruct] --subset [full|hard] --samples samples-sanitized-calibrated.jsonl
# ...If you really don't want to check the ground truths
bigcodebench.evaluate --split [complete|instruct] --subset [full|hard] --samples samples-sanitized-calibrated.jsonl --no-gt

# You are strongly recommended to use the following command to clean up the environment after evaluation:
pids=$(ps -u $(id -u) -o pid,comm | grep 'bigcodebench' | awk '{print $1}'); if [ -n \"$pids\" ]; then echo $pids | xargs -r kill; fi;
rm -rf /tmp/*
```

> [!Tip]
>
> Do you use a very slow machine?
>
> LLM solutions are regarded as **failed** on timeout (and OOM etc.).
> Specifically, we set the dynamic timeout based on the ground-truth solution's runtime.
>
> Additionally, you are **NOT** encouraged to make your test-bed over stressed while running evaluation.
> For example, using `--parallel 64` on a 4-core machine or doing something else during evaluation are bad ideas...

<details><summary>⌨️ More command-line flags <i>:: click to expand ::</i></summary>
<div>

* `--parallel`: by default half of the cores

</div>
</details>

The output should be like (below is GPT-4 greedy decoding example):

```
Asserting the groundtruth...
Expected outputs computed in 1200.0 seconds
Reading samples...
1140it [00:00, 1901.64it/s]
Evaluating samples...
100%|██████████████████████████████████████████| 1140/1140 [19:53<00:00, 6.75it/s]
BigCodeBench-Instruct-calibrated
Groundtruth pass rate: 1.000
pass@1: 0.568
```

- The "k" includes `[1, 5, 10]` where k values `<=` the sample size will be used
- A cache file named like `samples_eval_results.json` will be cached. Remove it to re-run the evaluation

<details><summary>🤔 How long it would take? <i>:: click to expand ::</i></summary>
<div>

If you do greedy decoding where there is only one sample for each task, the evaluation should take just a few minutes on Intel(R) Xeon(R) Gold 6150 CPU @ 2.70GHz, composed of 2 sockets, with 18 cores per socket. However, if you have multiple samples for each task, the evaluation will take longer.
Here are some tips to speed up the evaluation:

* Use `--parallel $(nproc)`
* Use our pre-evaluated results (see [LLM-generated code](#-LLM-generated-code))

</div>
</details>

## 🔍 Failure Inspection

You can inspect the failed samples by using the following command:

```bash
bigcodebench.inspect --eval-results sample-sanitized-calibrated_eval_results.json --in-place
```

## 🚀 Full Script

We provide a sample script to run the full pipeline:

```bash
bash run.sh
```

## 📊 Result Analysis

We provide a script to replicate the analysis like Elo Rating and Task Solve Rate, which helps you understand the performance of the models further.

```bash
To run the analysis, you need to put all the `samples_eval_results.json` files in a `results` folder, which is in the same directory as the script.

```bash
cd analysis
python get_results.py
```

## 💻 LLM-generated Code

We share pre-generated code samples from LLMs we have [evaluated](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard):
*  See the attachment of our [v0.1.5](https://github.com/bigcode-project/bigcodebench/releases/tag/v0.1.5). We include both `sanitized_samples.zip` and `sanitized_samples_calibrated.zip` for your convenience.

## 🐞 Known Issues

- [ ] Due to the flakes in the evaluation, the execution results may vary slightly (~0.2%) between runs. We are working on improving the evaluation stability.

- [ ] You may get errors like `ImportError: /usr/local/lib/python3.10/site-packages/matplotlib/_c_internal_utils.cpython-310-x86_64-linux-gnu.so: failed to map segment from shared object` when running the evaluation. This is due to the memory limit of the docker container. You can increase the memory limit of the docker container to solve this issue.

- [ ] We are aware of the issue of some users needing to use a proxy to access the internet. We are working on a subset of the tasks that do not require internet access to evaluate the code.

## 📜 Citation

```bibtex
@article{zhuo2024bigcodebench,
  title={BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions},
  author={Zhuo, Terry Yue and Vu, Minh Chien and Chim, Jenny and Hu, Han and Yu, Wenhao and Widyasari, Ratnadira and Yusuf, Imam Nur Bani and Zhan, Haolan and He, Junda and Paul, Indraneil and others},
  journal={arXiv preprint arXiv:2406.15877},
  year={2024}
}
```

## 🙏 Acknowledgement

- [EvalPlus](https://github.com/evalplus/evalplus)
