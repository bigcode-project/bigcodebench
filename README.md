# `OpenEval`


## 💻 LLM-generated code

We also share pre-generated code samples from LLMs we have [evaluated](https://open-eval.github.io/leaderboard.html):

Each sample file is packaged in a zip file named like `${model_name}_temp_${temperature}.zip`.
You can unzip them to a folder named like `${model_name}_temp_${temperature}` and run the evaluation from scratch with:

```bash
openeval.evaluate --dataset humaneval --samples ${model_name}_temp_${temperature}
```

## 🔨 Useful tools

To use these tools, please first install the repository from GitHub:

```bash
git clone https://github.com/openeval/openeval.git
cd openeval
pip install -r tools/requirements.txt
```

### Code generation

We have configured the code generation of a wide range of LLMs (see support details in [codegen/models.py](https://github.com/openeval/openeval/blob/master/codegen/model.py)).
Example to run greedy generation on StarCoderBase-7B:

```shell
python codegen/generate.py --model starcoderbase-7b --bs 1 --temperature 0 --n_samples 1 --resume --greedy --root [result_path] --dataset openeval
```

## 📜 Citation

```bibtex
```

## 🙏 Acknowledgement

- [EvalPlus](https://github.com/evalplus/evalplus)
