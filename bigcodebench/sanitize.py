"""Post-processing LLM-generated Python code implemented using tree-sitter."""

import os
import pathlib
from typing import Dict, Generator, List, Optional, Set, Tuple
from pqdm.processes import pqdm

from tqdm import tqdm
from tree_sitter import Node
from tree_sitter_languages import get_parser

from bigcodebench.data import (
    get_bigcodebench,
    load_solutions,
    write_directory,
    write_jsonl,
)
from bigcodebench.syncheck import syntax_check, api_check

CLASS_TYPE = "class_definition"
FUNCTION_TYPE = "function_definition"
IMPORT_TYPE = ["import_statement", "import_from_statement"]
IDENTIFIER_TYPE = "identifier"
ATTRIBUTE_TYPE = "attribute"
RETURN_TYPE = "return_statement"
EXPRESSION_TYPE = "expression_statement"
ASSIGNMENT_TYPE = "assignment"


def code_extract(text: str) -> str:
    lines = text.split("\n")
    longest_line_pair = (0, 0)
    longest_so_far = 0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            current_lines = "\n".join(lines[i : j + 1])
            if syntax_check(current_lines):
                current_length = sum(1 for line in lines[i : j + 1] if line.strip())
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)

    return "\n".join(lines[longest_line_pair[0] : longest_line_pair[1] + 1])


def get_deps(nodes: List[Tuple[str, Node]]) -> Dict[str, Set[str]]:

    def dfs_get_deps(node: Node, deps: Set[str]) -> None:
        for child in node.children:
            if child.type == IDENTIFIER_TYPE:
                deps.add(child.text.decode("utf8"))
            else:
                dfs_get_deps(child, deps)

    name2deps = {}
    for name, node in nodes:
        deps = set()
        dfs_get_deps(node, deps)
        name2deps[name] = deps
    return name2deps


def get_function_dependency(entrypoint: str, call_graph: Dict[str, str]) -> Set[str]:
    queue = [entrypoint]
    visited = {entrypoint}
    while queue:
        current = queue.pop(0)
        if current not in call_graph:
            continue
        for neighbour in call_graph[current]:
            if not (neighbour in visited):
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


def get_definition_name(node: Node) -> str:
    for child in node.children:
        if child.type == IDENTIFIER_TYPE:
            return child.text.decode("utf8")


def traverse_tree(node: Node) -> Generator[Node, None, None]:
    cursor = node.walk()
    depth = 0

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                depth += 1
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent() or depth == 0:
            break
        else:
            depth -= 1


def has_return_statement(node: Node) -> bool:
    traverse_nodes = traverse_tree(node)
    for node in traverse_nodes:
        if node.type == RETURN_TYPE:
            return True
    return False


def sanitize(code: str, solution: Dict, entrypoint: Optional[str] = None) -> str:
    code = code_extract(code.strip())
    code_bytes = bytes(code, "utf8")
    parser = get_parser("python")
    tree = parser.parse(code_bytes)
    class_names = set()
    function_names = set()
    variable_names = set()
    reachable = set()
    
    root_node = tree.root_node
    import_nodes = []
    definition_nodes = []

    for child in root_node.children:
        if child.type in IMPORT_TYPE:
            # if subset != "tool":
            import_nodes.append(child)
        elif child.type == CLASS_TYPE:
            name = get_definition_name(child)
            if not (
                name in class_names or name in variable_names or name in function_names
            ):
                definition_nodes.append((name, child))
                class_names.add(name)
        elif child.type == FUNCTION_TYPE:
            name = get_definition_name(child)
            if not (
                name in function_names or name in variable_names or name in class_names
            ):
                # if name == entrypoint:
                #     task_func_found = True
                # if task_func_found:
                definition_nodes.append((name, child))
                function_names.add(name)
        elif (
            child.type == EXPRESSION_TYPE and child.children[0].type == ASSIGNMENT_TYPE
        ):
            subchild = child.children[0]
            name = get_definition_name(subchild)
            if not (
                name in variable_names or name in function_names or name in class_names
            ):
                # if task_func_found:
                definition_nodes.append((name, subchild))
                variable_names.add(name)

    if entrypoint:
        name2deps = get_deps(definition_nodes)
        reachable = get_function_dependency(entrypoint, name2deps)

    sanitized_output = b""

    for node in import_nodes:
        sanitized_output += code_bytes[node.start_byte : node.end_byte] + b"\n"

    for pair in definition_nodes:
        name, node = pair
        if entrypoint and not (name in reachable):
            continue
        sanitized_output += code_bytes[node.start_byte : node.end_byte] + b"\n"
        
    sanitized_output = sanitized_output[:-1].decode("utf8")
    
    # ad-hoc approach to remove unnecessary lines, but it works
    lines = sanitized_output.splitlines()
    outer_lines = []
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith(" "):
            break
        if not lines[i].startswith(" ") and entrypoint in lines[i]:
            outer_lines.append(i)
    if outer_lines:
        sanitized_output = "\n".join(lines[: outer_lines[-1]])
    # if subset == "tool":
    #     return "" if api_check(solution[f"{split}_tool_"] + "\n" + sanitized_output) else sanitized_output
    # else:
    return sanitized_output


def process_solution(
    sample_solution: Dict,
    dataset: Dict,
    entry_point: Dict,
    subset: str,
    debug_task: str = None,
    calibrate: bool = False,
    is_folder: bool = False,
    target_path: str = None,
):

    task_id = sample_solution.get("task_id")
    if not task_id or task_id not in dataset:
        return None

    dbg_identifier = sample_solution["_identifier"]
    if debug_task is not None and task_id != debug_task:
        return None

    function_name = entry_point.get(task_id)
    old_code = sample_solution.get("solution")

    if old_code is None:
        assert "completion" in sample_solution, sample_solution
        old_code = dataset[task_id]["complete_prompt"] + "\n" + sample_solution.get("completion")
    else:
        if calibrate:
            old_code = old_code.replace("```python\n    ", "```python\n"+dataset[task_id]["complete_prompt"]+"    ")

    new_code = sanitize(code=old_code, solution=sample_solution, entrypoint=function_name)
    if subset == "tool":
        if api_check(new_code):
            new_code = ""
    # if old code and new code are different, print msg
    if new_code != old_code:
        msg = "Sanitized: " + dbg_identifier
        if is_folder:
            msg += " -> " + dbg_identifier.replace(samples, target_path)
        print(msg)

    return {"task_id": task_id, "solution": new_code}


def script(
    samples: str, subset: str, inplace: bool = False, debug_task: str = None, calibrate: bool = False, parallel: int=32
):
    # task_id -> entry_point
    entry_point = {}
    # merge two datasets
    dataset = {**get_bigcodebench(subset=subset)}

    for task_id, problem in dataset.items():
        entry_point[task_id] = problem["entry_point"]

    # make a new folder with "-sanitized" suffix
    is_folder = os.path.isdir(samples)
    target_path = pathlib.Path(samples)
    target_path_name = target_path.name
    if not inplace:
        if is_folder:
            if calibrate:
                target_path_name = target_path_name + "-sanitized-calibrated"
            else:
                target_path_name = target_path_name + "-sanitized"
        else:
            if calibrate:
                target_path_name = target_path_name.replace(".jsonl", "-sanitized-calibrated.jsonl")
            else:
                target_path_name = target_path_name.replace(".jsonl", "-sanitized.jsonl")
        target_path = target_path.parent / target_path_name
    target_path = str(target_path)

    nsan = 0
    ntotal = 0

    new_solutions = []

    parallel_arg_list = [
        {
            "sample_solution": sample_solution,
            "dataset": dataset,
            "entry_point": entry_point,
            "subset": subset,
            "debug_task": debug_task,
            "calibrate": calibrate,
            "is_folder": is_folder,
            "target_path": target_path
        }
        for sample_solution in load_solutions(samples)
    ]

    results = pqdm(parallel_arg_list, process_solution, n_jobs=min(parallel, os.cpu_count()), argument_type="kwargs")

    for result in results:
        if result is not None:
            print(result)
            new_solutions.append(result)
            nsan += 1
        ntotal += 1

    if is_folder:
        write_directory(target_path, new_solutions)
    else:
        write_jsonl(target_path, new_solutions)

    if nsan > 0:
        print(f"Sanitized {nsan} out of {ntotal} files.")
    else:
        print(f"All files seems valid -- no files are sanitized.")
    print(f"Check the sanitized files at {target_path}")


def main():
    from fire import Fire

    Fire(script)


if __name__ == "__main__":
    main()