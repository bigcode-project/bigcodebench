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
    <a href="#-news">📰 News</a> •
    <a href="#-quick-start">🔥 Quick Start</a> •
    <a href="#-remote-evaluation">🚀 Remote Evaluation</a> •
    <a href="#-llm-generated-code">💻 LLM-generated Code</a> •
    <a href="#-citation">📜 Citation</a>
</p>

## 📰 News
- **[2024-10-06]** We are releasing `bigcodebench==v0.2.0`!
- **[2024-10-05]** We create a public code execution API on the [Hugging Face space](https://huggingface.co/spaces/bigcode/bigcodebench-evaluator).
- **[2024-10-01]** We have evaluated 139 models on BigCodeBench-Hard so far. Take a look at the [leaderboard](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard)!
- **[2024-08-19]** To make the evaluation fully reproducible, we add a real-time code execution session to the leaderboard. It can be viewed [here](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard).
- **[2024-08-02]** We release `bigcodebench==v0.1.9`.

<details><summary>More News <i>:: click to expand ::</i></summary>
<div>

- **[2024-07-18]** We announce a subset of BigCodeBench, BigCodeBench-Hard, which includes 148 tasks that are more aligned with the real-world programming tasks. The details are available [in this blog post](https://huggingface.co/blog/terryyz/bigcodebench-hard). The dataset is available [here](https://huggingface.co/datasets/bigcode/bigcodebench-hard). The new release is `bigcodebench==v0.1.8`.
- **[2024-06-28]** We release `bigcodebench==v0.1.7`.
- **[2024-06-27]** We release `bigcodebench==v0.1.6`.
- **[2024-06-19]** We start the Hugging Face BigCodeBench Leaderboard! The leaderboard is available [here](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard).
- **[2024-06-18]** We release BigCodeBench, a new benchmark for code generation with 1140 software-engineering-oriented programming tasks. Preprint is available [here](https://arxiv.org/abs/2406.15877). PyPI package is available [here](https://pypi.org/project/bigcodebench/) with the version `0.1.5`.

</div>
</details>

## 🌸 About

### BigCodeBench

BigCodeBench is an **_easy-to-use_** benchmark for solving **_practical_** and **_challenging_** tasks via code. It aims to evaluate the true programming capabilities of large language models (LLMs) in a more realistic setting. The benchmark is designed for HumanEval-like function-level code generation tasks, but with much more complex instructions and diverse function calls.

### Why BigCodeBench?

BigCodeBench focuses on task automation via code generation with *diverse function calls* and *complex instructions*, with:

* ✨ **Precise evaluation & ranking**: See [our leaderboard](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard) for latest LLM rankings before & after rigorous evaluation.
* ✨ **Pre-generated samples**: BigCodeBench accelerates code intelligence research by open-sourcing [LLM-generated samples](#-LLM-generated-code) for various models -- no need to re-run the expensive benchmarks!

## 🔥 Quick Start

To get started, please first set up the environment:

```bash
# By default, you will use the remote evaluation API to execute the output samples.
pip install bigcodebench --upgrade

# You are suggested to use `flash-attn` for generating code samples.
pip install packaging ninja
pip install flash-attn --no-build-isolation
# Note: if you have installation problem, consider using pre-built
# wheels from https://github.com/Dao-AILab/flash-attention/releases
```

<details><summary>⏬ Install nightly version <i>:: click to expand ::</i></summary>
<div>

```bash
# Install to use bigcodebench.generate
pip install "git+https://github.com/bigcode-project/bigcodebench.git" --upgrade
```

</div>
</details>


## 🚀 Remote Evaluation

We use the greedy decoding as an example to show how to evaluate the generated code samples via remote API.
> [!Warning]
>
> To ease the generation, we use batch inference by default. However, the batch inference results could vary from *batch sizes to batch sizes* and *versions to versions*, at least for the vLLM backend. If you want to get more deterministic results for greedy decoding, please set `--bs` to `1`. 

> [!Note]
>
> Remotely executing on `BigCodeBench-Full` typically takes 6-7 minutes, and on `BigCodeBench-Hard` typically takes 4-5 minutes.

```bash
bigcodebench.evaluate \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --split [complete|instruct] \
  --subset [full|hard] \
  --backend [vllm|openai|anthropic|google|mistral|hf]
```

- All the resulted files will be stored in a folder named `bcb_results`.
- The generated code samples will be stored in a file named `[model_name]--bigcodebench-[instruct|complete]--[backend]-[temp]-[n_samples]-sanitized_calibrated.jsonl`.
- The evaluation results will be stored in a file named `[model_name]--bigcodebench-[instruct|complete]--[backend]-[temp]-[n_samples]-sanitized_calibrated_eval_results.json`.
- The pass@k results will be stored in a file named `[model_name]--bigcodebench-[instruct|complete]--[backend]-[temp]-[n_samples]-sanitized_calibrated_pass_at_k.json`.

> [!Note]
>
> BigCodeBench uses different prompts for base and chat models.
> By default it is detected by `tokenizer.chat_template` when using `hf`/`vllm` as backend.
> For other backends, only chat mode is allowed.
>
> Therefore, if your base models come with a `tokenizer.chat_template`,
> please add `--direct_completion` to avoid being evaluated
> in a chat mode.

Access OpenAI APIs from [OpenAI Console](https://platform.openai.com/)
```bash
export OPENAI_API_KEY=<your_openai_api_key>
```

Access Anthropic APIs from [Anthropic Console](https://console.anthropic.com/)
```bash
export ANTHROPIC_API_KEY=<your_anthropic_api_key>
```

Access Mistral APIs from [Mistral Console](https://console.mistral.ai/)
```bash
export MISTRAL_API_KEY=<your_mistral_api_key>
```

Access Gemini APIs from [Google AI Studio](https://aistudio.google.com/)
```bash
export GOOGLE_API_KEY=<your_google_api_key>
```

## 💻 LLM-generated Code

We share pre-generated code samples from LLMs we have [evaluated](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard):
*  See the attachment of our [v0.2.0](https://github.com/bigcode-project/bigcodebench/releases/tag/v0.2.0). We include `sanitized_samples_calibrated.zip` for your convenience.

## Advanced Usage

Please refer to the [ADVANCED USAGE](https://github.com/bigcode-project/bigcodebench/blob/main/ADVANCED_USAGE.md) for more details.

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
