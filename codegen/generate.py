import argparse
import os
import json
from os import PathLike

from model import DecoderBase, make_model
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

def code_generate(
    args, model: DecoderBase, id_range=None, version="default"
):
    with Progress(
        TextColumn(
            f"{args.dataset} •" + "[progress.percentage]{task.percentage:>3.0f}%"
        ),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as p:
        if args.dataset == "openeval":
            from openeval.data import get_open_eval, write_jsonl

            dataset = get_open_eval()

        # create save_path if it doesn't exist, e.g., a/b.jsonl
        dirname = os.path.dirname(args.save_path)
        if not os.path.exists(dirname) and dirname != "":
            os.makedirs(dirname)

        for task_id, task in p.track(dataset.items()):
            if id_range is not None:
                id_num = int(task_id.split("/")[1])
                low, high = id_range
                if id_num < low or id_num >= high:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue

            p_name = task_id.replace("/", "_")
            
            # read the existing file if save_path exists
            if os.path.exists(args.save_path):
                with open(args.save_path, "r") as f:
                    existing_data = f.read().splitlines()
            log = f"Codegen: {p_name} @ {model}"
            n_existing = 0
            if args.resume:
                if os.path.exists(args.save_path):
                    n_existing = len([1 for line in existing_data if json.loads(line)["task_id"] == task_id])
                else:
                    n_existing = 0
                if n_existing > 0:
                    log += f" (resuming from {n_existing})"

            nsamples = args.n_samples - n_existing
            p.console.print(log)

            sidx = args.n_samples - nsamples
            while sidx < args.n_samples:
                model.dataset = args.dataset
                prompt = task["prompt"].strip()
                outputs = model.codegen(
                    prompt,
                    do_sample=not args.greedy,
                    num_samples=args.n_samples - sidx,
                )
                assert outputs, "No outputs from model!"
                if model.direct_completion:
                    samples = [
                        dict(
                            task_id=task_id,
                            solution=prompt+completion,
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
                write_jsonl(args.save_path, samples, append=True)
                sidx += len(outputs)
                

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument(
        "--dataset", required=True, type=str, choices=["openeval"]
    )
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    # id_range is list
    parser.add_argument("--id-range", default=None, nargs="+", type=int)
    parser.add_argument("--version", default="default", type=str)
    parser.add_argument("--save-path", required=True, type=str)
    args = parser.parse_args()

    if (args.greedy and (args.temperature != 0 or args.bs != 1 or args.n_samples != 1))\
        or (args.temperature == 0 and args.bs == 1 and args.n_samples == 1):
        args.greedy = True
        args.temperature = 0
        args.bs = 1
        args.n_samples = 1
        print("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0")

    if args.id_range is not None:
        assert len(args.id_range) == 2, "id_range must be a list of length 2"
        assert args.id_range[0] < args.id_range[1], "id_range must be increasing"
        args.id_range = tuple(args.id_range)
    
    args.model = args.model.lower()
    model = make_model(
        name=args.model,
        batch_size=args.bs,
        temperature=args.temperature,
        dataset=args.dataset,
    )
    
    code_generate(
        args, model=model, id_range=args.id_range, version=args.version
    )


if __name__ == "__main__":
    main()
