# `🌳WildCodeBench`

> [!WARNING] 
> The project is under active development. Please check back later for more updates.

> [!WARNING]
> WildCode framework currently only supports the Code2Code generation task. We are working on adding the NL2Code task based on NL instructions.

<p align="center">
    <a href="#-quick-start">🔥Quick Start</a> •
    <a href="#-llm-generated-code">💻LLM code</a> •
    <a href="#-useful-tools">🔨Tools</a> •
    <a href="#-citation">📜Citation</a> •
    <a href="#-acknowledgement">🙏Acknowledgement</a>
</p>

## About

### WildCodeBench

WildCodeBench is an igorous benchmark for code generation with realistic constraints in the wild. It aims to evaluate the true programming capabilities of large language models (LLMs) in a more realistic setting. The benchmark is designed for HumanEval-like function-level code generation tasks, but with much more fine-grained descriptions and diverse tool use.

### WildCode

To facilitate the evaluation of LLMs on WildCodeBench, we provide a Python package `wild-code` that includes the dataset, generation scripts, and evaluation scripts. The package is built on top of the [EvalPlus](https://github.com/evalplus/evalplus) framework, which is a flexible and extensible evaluation framework for code generation tasks.

### Why WildCode?

WildCode is a rigorous evaluation framework for LLM4Code, with:

* ✨ **Precise evaluation & ranking**: See [our leaderboard](https://wildcodebench.github.io/leaderboard.html) for latest LLM rankings before & after rigorous evaluation.
* ✨ **Pre-generated samples**: WildCode accelerates code intelligence research by open-sourcing [LLM-generated samples](#-LLM-generated-code) for various models -- no need to re-run the expensive benchmarks!


## 🔥 Quick Start

> [!Tip]
>
> WildCode ❤️ [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness)!
> WildCodeBench will be integrated to bigcode-evaluation-harness that you can also run it there!

To get started, please first setup the environment:

```shell
pip install wild-code --upgrade
```

<details><summary>⏬ Install nightly version <i>:: click to expand ::</i></summary>
<div>

```shell
pip install "git+https://github.com/bigcode-project/wild-code.git" --upgrade
```

</div>
</details>

<details><summary>⏬ Using WildCode as a local repo? <i>:: click to expand ::</i></summary>
<div>

```shell
git clone https://github.com/bigcode-project/wild-code.git
cd wild-code
export PYTHONPATH=$PYTHONPATH:$(pwd)
pip install -e .
```

</div>
</details>

### Code generation

To generate code samples from a model, you can use the following command:

```shell
wildcode.generate --model [model_name] --dataset wildcodebench --greedy --bs [bs] --temperature [temp] --n_samples [n_samples] --resume --backend [vllm|hf|openai]
```
The generated code samples will be sored in a file named `[model_name]--wildcodebench--[backend]-[temp]-[n_samples].jsonl`.

<details><summary>🤔 Structure of `problem`? <i>:: click to expand ::</i></summary>
<div>

* `task_id` is the identifier string for the task
* `entry_point` is name of the function
* `prompt` is the function signature with docstring
* `instruction` is the instruction for the task completion
+ `canonical_solution` is the ground-truth implementation (re-implemented to fix bugs in HumanEval)
+ `test` is the `unittest` test case

</div>
</details>

> [!Note]
>
> **Expected Schema of `[model_name]--wildcodebench--[backend]-[temp]-[n_samples].jsonl`**
>
> 1. `task_id`: Task ID, which are the keys of `get_[human_eval|mbpp]_plus()`
> 2. `solution` (optional): Self-contained solution (usually including the prompt)
>    * Example: `{"task_id": "HumanEval/?", "solution": "def f():\n    return 1"}`

### Code post-processing

LLM-generated text may not be compilable code for including natural language lines or incomplete extra code.
We provide a tool namely `wildcode.sanitize` to clean up the code:

```shell
# 💡 If you are storing codes in jsonl:
wildcode.sanitize --samples samples.jsonl
# Sanitized code will be produced to `samples-sanitized.jsonl`

# 💡 If you are storing codes in directories:
wildcode.sanitize --samples /path/to/vicuna-[??]b_temp_[??]
# Sanitized code will be produced to `/path/to/vicuna-[??]b_temp_[??]-sanitized`
```

<details><summary>🔎 Checking the compilability of post-processed code<i>:: click to expand ::</i></summary>
<div>

To double-check the post-processing results, you can use `wildcode.syncheck` to check the code validity before and after sanitization, which will print erroneous code snippets and why they are wrong:

```shell
# 💡 If you are storing codes in jsonl:
wildcode.syncheck --samples samples.jsonl --dataset [wildcodebench]

# 💡 If you are storing codes in directories:
wildcode.syncheck --samples /path/to/vicuna-[??]b_temp_[??] --dataset [wildcodebench]
```

</div>
</details>


### Code evaluation

You are strongly recommended to use a sandbox such as [docker](https://docs.docker.com/get-docker/):

```bash
```

...Or if you want to try it locally regardless of the risks ⚠️:

```bash
wildcode.evaluate --dataset [wildcodebench] --samples samples.jsonl
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
Expected outputs computed in 2400.0 seconds
Reading samples...
964it [30:04, 37.79it/s]
Evaluating samples...
100%|██████████████████████████████████████████| 964/964 [00:03<00:00, 44.75it/s]
Base
{'pass@1': 0.548}
```

- `Base` is the `pass@k` for the original HumanEval
- The "k" includes `[1, 10, 100]` where k values `<=` the sample size will be used
- A cache file named like `samples_eval_results.jsonl` will be cached. Remove it to re-run the evaluation

<details><summary>🤔 How long it would take? <i>:: click to expand ::</i></summary>
<div>

If you do greedy decoding where there is only one sample for each task, the evaluation should take just a few seconds.
When running 1 samples x 964 tasks x all tests, it can take around ??-?? minutes by using `--parallel 64` and `--test-details`.
Here are some tips to speed up the evaluation:

* Use `--parallel $(nproc)`
* Use our pre-evaluated results (see [LLM-generated code](#-LLM-generated-code))

</div>
</details>

## Failure Inspection

You can inspect the failed samples by using the following command:

```shell
wildcode.inspect --dataset $DATASET --eval-results sample-sanitized_eval_results.json --in-place
```

## Full Script

We provide a sample script to run the full pipeline:

```shell
bash run.sh
```

## 💻 LLM-generated code

We share pre-generated code samples from LLMs we have [evaluated](https://wildcodebench.github.io/leaderboard.html):

## Known Issues

- [ ] We notice that some tasks heavily use memory for scientific modeling during testing. It will lead to timeout issues on some machines. If you get an error message like `Check failed: ret == 0 (11 vs. 0)Thread creation via pthread_create() failed.` in Tensorflow, it is very likely due to the memory issue. Try to allocate more memory to the process or reduce the number of parallel processes.

## 📜 Citation

```bibtex
```

## 🙏 Acknowledgement

- [EvalPlus](https://github.com/evalplus/evalplus)
