import os
import json
import argparse

from bigcodebench.model import DecoderBase, make_model
from bigcodebench.data import get_bigcodebench, write_jsonl
from bigcodebench.sanitize import sanitize
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
    batch_size: int=-1,
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
        
        # create save_path if it doesn't exist, e.g., a/b.jsonl
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname) and dirname != "":
            os.makedirs(dirname)
            
        batch_prompts = []
        batch_task_ids = []
        batch_nsamples = []
        batch_entry_points = []
        
        # Read existing data once if resuming
        existing_data = {}
        if resume and os.path.exists(save_path):
            with open(save_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    existing_data[item["task_id"]] = existing_data.get(item["task_id"], 0) + 1
        
        for id_num, (task_id, task) in enumerate(p.track(dataset.items())):
            if id_range is not None:
                low, high = id_range
                if id_num < low:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue
                if id_num > id_range[1]:
                    break

            p_name = task_id.replace("/", "_")

            n_existing = existing_data.get(task_id, 0)
            nsamples = n_samples - n_existing
            
            try:
                prompt = task[f"{split}_prompt"]
            except:
                raise Exception(f"Invalid split {split} for bigcodebench-{subset}")
            if strip_newlines:
                prompt = prompt.strip("\n")
            
            if nsamples > 0:
                batch_prompts.append(prompt)
                batch_task_ids.append(task_id)
                batch_nsamples.append(nsamples)
                batch_entry_points.append(task["entry_point"])
                
                log = f"Codegen: {p_name} @ {model}"
                if n_existing > 0:
                    log += f" (resuming from {n_existing})"
                p.console.print(log)
            
            if (batch_size and len(batch_prompts) == batch_size) or id_num == len(dataset) - 1 or (id_range and id_num == id_range[1] - 1):
                if not batch_prompts and id_num == len(dataset) - 1:
                    break
                outputs = model.codegen(
                    batch_prompts,
                    do_sample=not greedy,
                    num_samples=max(batch_nsamples),
                )
                assert outputs, "No outputs from model!"
                
                samples = []
                for task_id, content, entry_point, nsamples, task_outputs in zip(batch_task_ids, batch_prompts, batch_entry_points, batch_nsamples, outputs):
                    if model.is_direct_completion():
                        samples.extend([
                            dict(task_id=task_id, solution=sanitize(content+completion, entry_point))
                            for completion in task_outputs[:nsamples]
                        ])
                    else:
                        samples.extend([
                            dict(task_id=task_id, solution=sanitize(completion, entry_point))
                            for completion in task_outputs[:nsamples]
                        ])
                print(f"Generated {len(samples)} samples")
                write_jsonl(save_path, samples, append=True)
            
                # Clear batches
                batch_prompts = []
                batch_task_ids = []
                batch_nsamples = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--split", required=True, type=str, choices=["complete", "instruct"])
    parser.add_argument("--subset", default="full", type=str, choices=["full", "hard"])
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--strip_newlines", action="store_true")
    parser.add_argument("--direct_completion", action="store_true")
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
        temperature=args.temperature,
        base_url=args.base_url,
        tp=args.tp,
        trust_remote_code=args.trust_remote_code,
        direct_completion=args.direct_completion,
        tokenizer_name=args.tokenizer_name,
        tokenizer_legacy=args.tokenizer_legacy
    )
    
    extra = "-" + args.subset if args.subset != "full" else ""
    if not args.save_path:
        save_path = args.model.replace("/", "--") + f"--bigcodebench{extra}-{args.split}--{args.backend}-{args.temperature}-{args.n_samples}-sanitized_calibrated.jsonl"
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
        id_range=args.id_range,
        batch_size=args.bs
    )


if __name__ == "__main__":
    main()
