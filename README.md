# BigCodeBench
<center>
<img src="https://github.com/bigcode-bench/bigcode-bench.github.io/blob/main/asset/bigcodebench_banner.png?raw=true" alt="BigCodeBench">
</center>

> [!WARNING]
> Please use BigCodeBench with caution. Different from [EvalPlus](https://github.com/evalplus/evalplus), BigCodeBench has a much less constrained execution environment to support tasks with diverse library dependencies. This may lead to security risks. We recommend using a sandbox such as [Docker](https://docs.docker.com/get-docker/) to run the evaluation.

<p align="center">
    <a href="https://pypi.org/project/bigcodebench/"><img src="https://img.shields.io/pypi/v/bigcodebench?color=g"></a>
    <a href="https://hub.docker.com/r/terryzho/bigcodebench-evaluate" title="Docker-Eval"><img src="https://img.shields.io/docker/image-size/terryzho/bigcodebench-evaluate"></a>
    <a href="https://hub.docker.com/r/terryzho/bigcodebench-generate-cu11" title="Docker-Gen-CU11"><img src="https://img.shields.io/docker/image-size/terryzho/bigcodebench-generate-cu11"></a>
    <a href="https://hub.docker.com/r/terryzho/bigcodebench-generate-cu12" title="Docker-Gen-CU12"><img src="https://img.shields.io/docker/image-size/terryzho/bigcodebench-generate-cu12"></a>
    <a href="https://github.com/bigcodebench/bigcodebench/blob/master/LICENSE"><img src="https://img.shields.io/pypi/l/bigcodebench"></a>
</p>

<p align="center">
    <a href="#-about">🌸About</a> •
    <a href="#-quick-start">🔥Quick Start</a> •
    <a href="#-llm-generated-code">💻LLM code</a> •
    <a href="#-failure-inspection">🔍Failure inspection</a> •
    <a href="#-known-issues">🐞Known issues</a> •
    <a href="#-citation">📜Citation</a> •
    <a href="#-acknowledgement">🙏Acknowledgement</a>
</p>

## About

### BigCodeBench

BigCodeBench is a rigorous benchmark for code generation with realistic constraints in the wild. It aims to evaluate the true programming capabilities of large language models (LLMs) in a more realistic setting. The benchmark is designed for HumanEval-like function-level code generation tasks, but with much more fine-grained descriptions and diverse tool use.
To facilitate the evaluation of LLMs on BigCodeBench, we provide a Python package `bigcodebench` that includes the dataset, generation scripts, and evaluation scripts. The package is built on top of the [EvalPlus](https://github.com/evalplus/evalplus) framework, which is a flexible and extensible evaluation framework for code generation tasks.

### Why BigCodeBench?

BigCodeBench focuses on the evaluation of LLM4Code with *diverse function calls* and *complex instruction*, with:

* ✨ **Precise evaluation & ranking**: See [our leaderboard](https://bigcodebench.github.io/leaderboard.html) for latest LLM rankings before & after rigorous evaluation.
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

```shell
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

```shell
pip install "git+https://github.com/bigcode-project/bigcodebench.git" --upgrade
```

</div>
</details>

<details><summary>⏬ Using BigCodeBench as a local repo? <i>:: click to expand ::</i></summary>
<div>

```shell
git clone https://github.com/bigcode-project/bigcodebench.git
cd bigcodebench
export PYTHONPATH=$PYTHONPATH:$(pwd)
pip install -e .
```

</div>
</details>

### Code Generation

You are suggested to use `flash-attn` for generating code samples.
```shell
pip install -U flash-attn
```

To generate code samples from a model, you can use the following command:
>
```shell
bigcodebench.generate \
    --model [model_name] \
    --subset [complete|instruct] \
    --greedy \
    --bs [bs] \
    --temperature [temp] \
    --n_samples [n_samples] \
    --resume \
    --backend [vllm|hf|openai|mistral|anthropic|google] \
    --tp [gpu_number]
```
>
The generated code samples will be stored in a file named `[model_name]--bigcodebench-[instruct|complete]--[backend]-[temp]-[n_samples].jsonl`. Alternatively, you can use the following command to utilize our pre-built docker images for generating code samples:
>
```shell
docker run --gpus '"device=$CUDA_VISIBLE_DEVICES"' -v $(pwd):/bigcodebench -t terryzho/bigcodebench-generate-cu11:latest \
    --model [model_name] \ 
    --subset [complete|instruct] \
    --greedy \
    --bs [bs] \   
    --temperature [temp] \
    --n_samples [n_samples] \
    --resume \
    --backend [vllm|hf|openai|mistral|anthropic|google] \
    --tp [gpu_number]
```
>
We make available `cuda 11.8.0` and `cuda 12.1.1` pre-built docker images with the Dockerfiles available in the `Docker` directory.
>
If you wish to use gated or private HuggingFace models and datasets, you need to build the container yourself with `--build-arg` flags as follows:
>
```shell
docker build --build-arg HF_TOKEN=<YOUR_HF_TOKEN> -t terryzho/bigcodebench-generate-cu11:latest - < Docker/Generate_Cuda11.Dockerfile
```
>
Following which, you can run the built container as shown in above.
>
<details><summary>🤔 Structure of `problem`? <i>:: click to expand ::</i></summary>
<div>

* `task_id` is the identifier string for the task
* `entry_point` is the name of the function
* `prompt` is the function signature with docstring
* `instruction` is the instruction for the task completion
+ `canonical_solution` is the ground-truth implementation
+ `test` is the `unittest` test case

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

```shell
# 💡 If you want to store calibrated codes in jsonl for evaluation:
bigcodebench.sanitize --samples samples.jsonl --calibrate
# Sanitized code will be produced to `samples-sanitized-calibrated.jsonl`

# 💡 If you are doing without calibration:
bigcodebench.sanitize --samples samples.jsonl
# Sanitized code will be produced to `samples-sanitized.jsonl`

# 💡 If you are storing codes in directories:
bigcodebench.sanitize --samples /path/to/vicuna-[??]b_temp_[??]
# Sanitized code will be produced to `/path/to/vicuna-[??]b_temp_[??]-sanitized`
```

<details><summary>🔎 Checking the compatibility of post-processed code<i>:: click to expand ::</i></summary>
<div>

To double-check the post-processing results, you can use `bigcodebench.syncheck` to check the code validity before and after sanitization, which will print erroneous code snippets and why they are wrong:

```shell
# 💡 If you are storing codes in jsonl:
bigcodebench.syncheck --samples samples-calibrated.jsonl --dataset [bigcodebench]

# 💡 If you are storing codes in directories:
bigcodebench.syncheck --samples /path/to/vicuna-[??]b_temp_[??] --dataset [bigcodebench]
```

</div>
</details>


### Code Evaluation

You are strongly recommended to use a sandbox such as [docker](https://docs.docker.com/get-docker/):

```shell
# mount the current directory to the container
docker run -v $(pwd):/bigcodebench terryzho/bigcodebench-evaluate:latest --subset [complete|instruct] --samples samples-calibrated.jsonl
```

...Or if you want to try it locally regardless of the risks ⚠️:

First, install the dependencies for BigCodeBench:

```shell
pip install -r https://raw.githubusercontent.com/bigcode-project/bigcodebench-annotation/main/requirements.txt
```

Then, run the evaluation:

```shell
# ...Or locally ⚠️
bigcodebench.evaluate --subset [complete|instruct] --samples samples-calibrated.jsonl
# ...If the ground truth is not working locally
bigcodebench.evaluate --subset [complete|instruct] --samples samples-calibrated.jsonl --no-gt
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
bigcodebench
{'pass@1': 0.568}
```

- The "k" includes `[1, 5, 10]` where k values `<=` the sample size will be used
- A cache file named like `samples_eval_results.jsonl` will be cached. Remove it to re-run the evaluation

<details><summary>🤔 How long it would take? <i>:: click to expand ::</i></summary>
<div>

If you do greedy decoding where there is only one sample for each task, the evaluation should take just a few seconds.
When running 1 sample x 964 tasks x all tests, it can take around ??-?? minutes by using `--parallel 64` and `--test-details`.
Here are some tips to speed up the evaluation:

* Use `--parallel $(nproc)`
* Use our pre-evaluated results (see [LLM-generated code](#-LLM-generated-code))

</div>
</details>

## Failure Inspection

You can inspect the failed samples by using the following command:

```shell
bigcodebench.inspect --dataset [bigcodebench] --eval-results sample-sanitized_eval_results.json --in-place
```

## Full script

We provide a sample script to run the full pipeline:

```shell
bash run.sh
```

## 💻 LLM-generated Code

We share pre-generated code samples from LLMs we have [evaluated](https://bigcodebench.github.io/leaderboard.html):
*  See the attachment of our [v0.1.2](https://github.com/bigcode-project/bigcodebench/releases/tag/v0.1.2). We include both `sanitized_samples.zip` and `sanitized_samples_calibrated.zip` for your convenience.

## Known Issues

- [ ] We notice that some tasks heavily use memory for scientific modeling during testing. It will lead to timeout issues on some machines. If you get an error message like `Check failed: ret == 0 (11 vs. 0)Thread creation via pthread_create() failed.` in Tensorflow, it is very likely due to the memory issue. Try to allocate more memory to the process or reduce the number of parallel processes.

- [ ] Due to the flakes in the evaluation, the execution results may vary slightly (~0.5%) between runs. We are working on improving the evaluation stability.

- [ ] We are aware of the issue that some users may need to use a proxy to access the internet. We are working on a subset of the tasks that do not require internet access to evaluate the code.

## 📜 Citation

```bibtex
@article{bigcodebench,
  title={BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions},
  author={Zhuo, Terry Yue and Vu, Min Chien and Chim, Jenny and Hu, Han and Yu, Wenhao and Widyasari, Ratnadira and Yusuf, Imam Nur Bani and Zhan, Haolan and He, Junda and Paul, Indraneil and Brunner, Simon and Gong, Chen and Hoang, Thong and Zebaze, Armel Randy and Hong, Xiaoheng and Li, Wen-Ding and Kaddour, Jean and Xu, Ming and Zhang, Zhihan and Yadav, Prateek and Jain, Naman and Gu, Alex and Cheng, Zhoujun and Liu, Jiawei and Liu, Qian and Wang, Zijian and Lo, David and Hui, Binyuan and Muennighoff, Niklas and Fried, Daniel and Du, Xiaoning and de Vries, Harm and Von Werra, Leandro},
  year={2024}
}
```

## 🙏 Acknowledgement

- [EvalPlus](https://github.com/evalplus/evalplus)
