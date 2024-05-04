"""Post-processing LLM-generated Python code implemented using tree-sitter."""

import os
import ast
import astunparse
import pathlib
from typing import Dict, Generator, List, Optional, Set, Tuple

from tqdm import tqdm
from tree_sitter import Node
from tree_sitter_languages import get_parser

from openeval.data import (
    get_open_eval,
    load_solutions,
    write_directory,
    write_jsonl,
)
from openeval.syncheck import syntax_check

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


def get_callee_name(
    node: Node, class_names: Set[str], function_names: Set[str]
) -> Optional[str]:
    for child in node.children:
        if child.type == ATTRIBUTE_TYPE:
            name = child.children[0].text.decode("utf8")
            if name in class_names:
                return name
        elif child.type == IDENTIFIER_TYPE:
            name = child.text.decode("utf8")
            if name in function_names or name in class_names:
                return name


def get_call_graph(
    nodes: List[Tuple[str, Node]], class_names: Set[str], function_names: Set[str]
) -> Dict[str, str]:
    call_graph = {}
    for name, node in nodes:
        function_calls = []
        traverse_nodes = traverse_tree(node)
        for node in traverse_nodes:
            if node.type == "call":
                callee_name = get_callee_name(node, class_names, function_names)
                if callee_name:
                    function_calls.append(callee_name)
        call_graph[name] = function_calls
    return call_graph


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


def extract_entry_code(code, function_name):
    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Retrieve the function's docstring
                docstring = ast.get_docstring(node)
                if docstring:
                    # Find the position just after the docstring node
                    if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                        docstring_node = node.body[0]
                        # Get the line number where the docstring ends
                        docstring_end_line = docstring_node.end_lineno
                        lines = code.splitlines()
                        function_code = "\n".join(lines[docstring_end_line:node.end_lineno])
                        return function_code
                else:
                    return astunparse.unparse(node.body)
    except:
        pass

    return code


def sanitize(code: str, entrypoint: Optional[str] = None) -> str:
    code = code_extract(code)
    parser = get_parser("python")
    tree = parser.parse(bytes(code, "utf8"))
    class_names = set()
    function_names = set()
    variable_names = set()

    root_node = tree.root_node
    import_nodes = []
    definition_nodes = []
    
    entry_end_byte = -1
    for child in root_node.children:
        if child.type in IMPORT_TYPE:
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
                # hard-code the special case for the entrypoint
                if name == entrypoint:
                    entry_end_byte = child.end_byte
                    entry_code = code[child.start_byte:child.end_byte]
                    if "\n\n#" in entry_code:
                        entry_end_byte = entry_code.index("\n\n#") + child.start_byte
                definition_nodes.append((name, child))
                function_names.add(get_definition_name(child))
        elif (
            child.type == EXPRESSION_TYPE and child.children[0].type == ASSIGNMENT_TYPE
        ):
            subchild = child.children[0]
            name = get_definition_name(subchild)
            if not (
                name in variable_names or name in function_names or name in class_names
            ):
                definition_nodes.append((name, subchild))
                variable_names.add(name)

    if entrypoint:
        call_graph = get_call_graph(definition_nodes, class_names, function_names)
        reacheable = get_function_dependency(entrypoint, call_graph)

    sanitized_output = ""
    for node in import_nodes:
        sanitized_output += code[node.start_byte : node.end_byte] + "\n"

    for pair in definition_nodes:
        name, node = pair
        if not (name in variable_names) and entrypoint and not (name in reacheable):
            continue
        if node.start_byte < entry_end_byte or entry_end_byte == -1:
            if node.end_byte > entry_end_byte and entry_end_byte:
                sanitized_output += code[node.start_byte : entry_end_byte] + "\n"
            else:
                sanitized_output += code[node.start_byte : node.end_byte] + "\n"
    # print(extract_entry_code(sanitized_output[:-1], entrypoint))
    # return sanitized_output[:-1]
    return extract_entry_code(sanitized_output[:-1], entrypoint)


def script(
    samples: str, inplace: bool = False, debug_task: str = None
):
    # task_id -> entry_point
    entry_point = {}
    # merge two datasets
    dataset = {**get_open_eval()}

    for task_id, problem in dataset.items():
        entry_point[task_id] = problem["entry_point"]

    # make a new folder with "-sanitized" suffix
    is_folder = os.path.isdir(samples)
    target_path = pathlib.Path(samples)
    if not inplace:
        if is_folder:
            new_name = target_path.name + "-sanitized-plus"
        else:
            new_name = target_path.name.replace(".jsonl", "-sanitized-plus.jsonl")
        target_path = target_path.parent / new_name
    target_path = str(target_path)

    nsan = 0
    ntotal = 0

    new_solutions = []

    for solution in tqdm(load_solutions(samples)):
        task_id = solution["task_id"]
        if task_id not in dataset:
            print(
                f"Skiping {task_id} as it does not existing in the latest EvalPlus dataset."
            )
            continue

        function_name = entry_point[task_id] if task_id in entry_point else None
        dbg_identifier = solution["_identifier"]
        if debug_task is not None and task_id != debug_task:
            continue

        ntotal += 1
        if "solution" in solution:
            old_code = solution["solution"]
        else:
            assert "completion" in solution
            old_code = dataset[task_id]["prompt"] + "\n" + solution["completion"]

        new_code = dataset[task_id]["prompt"] + "\n" + sanitize(code=old_code, entrypoint=function_name)
        # if changed, print the message
        if new_code != old_code:
            msg = "Sanitized: " + dbg_identifier
            if is_folder:
                msg += " -> " + dbg_identifier.replace(samples, target_path)
            print(msg)
            nsan += 1

        new_solutions.append({"task_id": task_id, "solution": new_code})

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
