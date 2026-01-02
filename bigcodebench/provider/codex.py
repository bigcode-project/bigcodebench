import os
import tempfile
import subprocess
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from bigcodebench.provider.base import DecoderBase


class CodexDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.model = name

    def codegen(
        self, prompts: List[str], do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        # For each prompt, run codex in parallel
        with ProcessPoolExecutor(max_workers=len(prompts)) as executor:
            futures = [executor.submit(self._run_codex, i, prompt, num_samples) for i, prompt in enumerate(prompts)]
            # Use tqdm to display progress as workers complete so the caller
            # (generate.py) sees progress during long-running Codex jobs.
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Codex: {self.model}"):
                results.append(future.result())
        # Sort results back to original order
        results.sort(key=lambda x: x[0])
        return [res[1] for res in results]

    def _run_codex(self, idx: int, prompt: str, num_samples: int) -> tuple[int, List[str]]:
        # Since num_samples is per prompt, but codex might generate one, we assume num_samples=1 for now
        # If codex can generate multiple, adjust
        # For simplicity, assume one per run
        outputs = []
        for _ in range(num_samples):
            with tempfile.TemporaryDirectory() as temp_dir:
                # Modify prompt to instruct writing to solution.py
                modified_prompt = prompt + "\n\nWrite the complete solution code to the file 'solution.py' in the current directory."
                # Run codex
                try:
                    result = subprocess.run(
                        ["codex", "exec", "--skip-git-repo-check", "--cd", temp_dir,"--full-auto", modified_prompt],
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 min timeout
                    )
                    if result.returncode == 0:
                        solution_file = os.path.join(temp_dir, "solution.py")
                        if os.path.exists(solution_file):
                            with open(solution_file, "r") as f:
                                code = f.read()
                            outputs.append(code)
                        else:
                            outputs.append("")  # No file
                    else:
                        outputs.append("")  # Error
                except subprocess.TimeoutExpired:
                    outputs.append("")
        return (idx, outputs)

    def is_direct_completion(self) -> bool:
        return False