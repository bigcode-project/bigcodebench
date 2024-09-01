"""This file checks two things:
1. Is the LLMs codegen completed for each benchmark?
2. Warn the code that are not compilable (it could be some impl issues).
"""

import ast
import traceback

from termcolor import colored

from bigcodebench.data import load_solutions

def api_check(code: str) -> bool:
    tree = ast.parse(code)
    imported_modules = set()
    imported_names = {}

    class ApiExtractor(ast.NodeVisitor):
        def __init__(self):
            self.in_task_func = False
            self.uses_library_api = False

        def visit_Import(self, node):
            for alias in node.names:
                imported_modules.add(alias.name)
                if alias.asname:
                    imported_modules.add(alias.asname)

        def visit_ImportFrom(self, node):
            if node.module:
                for alias in node.names:
                    full_name = f'{node.module}.{alias.name}'
                    imported_names[alias.asname or alias.name] = full_name

        def visit_FunctionDef(self, node):
            if node.name == 'task_func':
                self.in_task_func = True
                self.generic_visit(node)
                self.in_task_func = False
            else:
                self.generic_visit(node)

        def visit_Attribute(self, node):
            if self.in_task_func:
                attr_chain = []
                current = node
                while isinstance(current, ast.Attribute):
                    attr_chain.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    attr_chain.append(current.id)
                    attr_chain.reverse()
                    full_name = '.'.join(attr_chain)
                    if attr_chain[0] in imported_modules or attr_chain[0] in imported_names:
                        self.uses_library_api = True
            self.generic_visit(node)

        def visit_Name(self, node):
            if self.in_task_func:
                if node.id in imported_modules or node.id in imported_names:
                    self.uses_library_api = True
            self.generic_visit(node)

    extractor = ApiExtractor()
    extractor.visit(tree)

    return extractor.uses_library_api

def syntax_check(code, verbose=False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


def script(
    samples: str, nsample_check: int = None, verbose: bool = False
):
    # List[Dict{"task_id", "solution"}]
    solutions = load_solutions(samples)

    from bigcodebench.data import get_bigcodebench

    dataset = get_bigcodebench()
    dataset_name = "BigCodeBench"

    print(colored(f"Dataset: {dataset_name}", "blue"))

    id2solutions = {}
    for solution in solutions:
        task_id = solution["task_id"]
        if task_id not in id2solutions:
            id2solutions[task_id] = []
        if "solution" not in solution:
            assert "completion" in solution, "solution or completion must exist!"
            solution["solution"] = dataset[task_id]["complete_prompt"] + solution["completion"]
        id2solutions[task_id].append(solution)

    print(colored("==============================", "blue"))
    print(colored(" ::: Checking completeness... ", "blue"))
    print(colored(" ::::: All tasks complete?    ", "blue"))
    ndone = 0

    task_ids = dataset.keys()
    ntask = len(task_ids)
    for task_id in task_ids:
        if task_id not in id2solutions:
            print(colored(f" ⚠️ {task_id} is missing!", "red"))
            continue
        nfiles = len(id2solutions[task_id])

        if nsample_check is None or nfiles <= nsample_check:
            ndone += 1
            continue

        print(
            colored(
                f" ⚠️ {task_id} only has {nfiles} samples! But {nsample_check} are expected.",
                "red",
            )
        )

    # check if there is enough number of samples here.
    if nsample_check is not None:
        if ntask != ndone:
            ntbd = ntask - ndone
            print(colored(f" ::::: ⚠️ {ntbd}/{ntask} tasks incomplete!", "red"))
        else:
            print(colored(f" ::::: All {ntask} tasks complete!", "green"))

    print(colored("==============================", "blue"))
    print(colored(" ::: Checking compilation...  ", "blue"))
    print(colored(" ::::: All code compilable?   ", "blue"))
    ncode = 0
    nwrong = 0
    for task_id in task_ids:
        # task_id must exist
        if task_id not in id2solutions:
            continue

        for solution in id2solutions[task_id]:
            ncode += 1
            code = solution["solution"]
            dbg_identifier = solution["_identifier"]
            if code.strip() == "":
                print(colored(f" ⚠️ {dbg_identifier} is empty!", "red"))
                nwrong += 1
            elif not syntax_check(code, verbose):
                print(colored(f" ⚠️ {dbg_identifier} is not compilable!", "red"))
                nwrong += 1
    if 0 != nwrong:
        print(colored(f" ::::: ⚠️ {nwrong}/{ncode} code are not compilable!", "red"))
    else:
        print(colored(f" ::::: All {ncode} code are compilable!", "green"))


def main():
    from fire import Fire

    Fire(script)


if __name__ == "__main__":
    main()
