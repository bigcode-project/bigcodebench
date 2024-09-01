import os
import json
import argparse

from bigcodebench.model import DecoderBase, make_model
from bigcodebench.data import get_bigcodebench, write_jsonl
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)


def codegen(
    model: DecoderBase,
    save_path: str,
    split: str,
    subset="full",
    greedy=False,
    strip_newlines=False,
    n_samples=1,
    id_range=None,
    resume=True,
):
    with Progress(
        TextColumn(f"BigCodeBench--{split.capitalize()} ({subset.capitalize()}) •" + "[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as p:
            
        dataset = get_bigcodebench(subset=subset)

        if model.is_direct_completion() and split == "instruct":
            raise Exception("Base model does not support direct completion for instruct tasks")
        
        if subset == "tool":
            assert split in ["positive", "negative", "mixed"], "Tool subset only supports positive, negative, and mixed split"
        # create save_path if it doesn't exist, e.g., a/b.jsonl
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname) and dirname != "":
            os.makedirs(dirname)
        for id_num, (task_id, task) in enumerate(p.track(dataset.items())):
            if id_range is not None:
                low, high = id_range
                if id_num < low or id_num >= high:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue

            p_name = task_id.replace("/", "_")

            # read the existing file if save_path exists
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    existing_data = f.read().splitlines()
            log = f"Codegen: {p_name} @ {model}"
            n_existing = 0
            if resume:
                if os.path.exists(save_path):
                    n_existing = len([1 for line in existing_data if json.loads(line)["task_id"] == task_id])
                else:
                    n_existing = 0
                if n_existing > 0:
                    log += f" (resuming from {n_existing})"

            nsamples = n_samples - n_existing
            p.console.print(log)

            sidx = n_samples - nsamples
            while sidx < n_samples:
                try:
                    if subset == "tool":
                        prompt = task[f"{split}_tool"] + "\n\n" + task["complete_prompt"]
                    else:
                        prompt = task[f"{split}_prompt"]
                except:
                    raise Exception(f"Invalid split {split} for bigcodebench-{subset}")
                if strip_newlines:
                    prompt = prompt.strip("\n")
                outputs = model.codegen(
                    prompt,
                    do_sample=not greedy,
                    num_samples=n_samples - sidx,
                )
                assert outputs, "No outputs from model!"
                if model.is_direct_completion():
                    samples = [
                        dict(
                            task_id=task_id,
                            solution=task["complete_prompt"]+completion
                        )
                        for task_id, completion in zip([task_id]*len(outputs), outputs)
                    ]
                else:
                    samples = [
                        dict(
                            task_id=task_id,
                            solution=completion,
                        )
                        for task_id, completion in zip([task_id]*len(outputs), outputs)
                    ]
                print(f"Generated {len(samples)} samples")
                write_jsonl(save_path, samples, append=True)
                sidx += len(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--split", required=True, type=str, choices=["complete", "instruct", "positive", "negative", "mixed"])
    parser.add_argument("--subset", default="full", type=str, choices=["full", "hard", "tool"])
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--strip_newlines", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--id_range", nargs=2, type=int)
    parser.add_argument("--backend", default="vllm", type=str, choices=["vllm", "hf", "openai", "mistral", "anthropic", "google"])
    parser.add_argument("--base_url", default=None, type=str)
    parser.add_argument("--tp", default=1, type=int)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--tokenizer_legacy", action="store_true")
    parser.add_argument("--tokenizer_name", default=None, type=str)

    args = parser.parse_args()

    if args.greedy or (args.temperature == 0 and args.n_samples == 1):
        args.temperature = 0
        args.bs = 1
        args.n_samples = 1
        args.greedy = True
        print("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0")

    if args.id_range is not None:
        assert len(args.id_range) == 2, "id_range must be a list of length 2"
        assert args.id_range[0] < args.id_range[1], "id_range must be increasing"
        args.id_range = tuple(args.id_range)

    # Make dir for codes generated by each model
    model_runner = make_model(
        model=args.model,
        backend=args.backend,
        subset=args.subset,
        split=args.split,
        batch_size=args.bs,
        temperature=args.temperature,
        base_url=args.base_url,
        tp=args.tp,
        trust_remote_code=args.trust_remote_code,
        tokenizer_name=args.tokenizer_name,
        tokenizer_legacy=args.tokenizer_legacy
    )
    
    extra = "-" + args.subset if args.subset != "full" else ""
    if not args.save_path:
        save_path = args.model.replace("/", "--") + f"--bigcodebench{extra}-{args.split}--{args.backend}-{args.temperature}-{args.n_samples}.jsonl"
    else:
        save_path = args.save_path

    codegen(
        model=model_runner,
        save_path=save_path,
        split=args.split,
        subset=args.subset,
        greedy=args.greedy,
        strip_newlines=args.strip_newlines,
        n_samples=args.n_samples,
        resume=args.resume,
        id_range=args.id_range
    )


if __name__ == "__main__":
    main()
