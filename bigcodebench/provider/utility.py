from typing import List

EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
]


def extra_eos_for_direct_completion(dataset) -> List[str]:
    if dataset.lower() == "bigcodebench":
        return ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
    raise ValueError(f"Unknown dataset: {dataset}")


# some random words which serves as the splitter
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"


def make_raw_chat_prompt(
    task_prompt: str,
    subset: str,
    split: str, 
    instruction_prefix: str,
    response_prefix: str,
    tokenizer: AutoTokenizer,
    direct_completion: bool = False,
) -> str:
    # directly return prompt if it does not have a tokenizer.chat_template
    if tokenizer:
        if tokenizer.chat_template is None or direct_completion:
            return task_prompt

    assert instruction_prefix is not None, "Instruction prefix is required!"
    assert response_prefix is not None, "Response prefix is required!"
    
    if split == "complete":
        task_prompt = f"""\
{instruction_prefix}
```
{task_prompt.strip()}
```
"""
    else:
        task_prompt = f"""\
{instruction_prefix}
{task_prompt.strip()}
"""
    response = f"""\
{response_prefix}
```python
{_MAGIC_SPLITTER_}
```
"""
    task_prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
    ).split(_MAGIC_SPLITTER_)[0]
    return task_prompt