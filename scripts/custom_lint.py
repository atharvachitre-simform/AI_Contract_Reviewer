import ast
import os
import sys


def lint_file(filepath):
    warnings = []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. File LOC limit
    lines = content.splitlines()
    if len(lines) > 500:
        warnings.append(f"WARNING: File exceeds 500 LOC ({len(lines)} lines)")

    try:
        tree = ast.parse(content, filename=filepath)
    except SyntaxError as e:
        warnings.append(f"ERROR: Syntax error during parsing: {e}")
        return warnings

    class ComplexityVisitor(ast.NodeVisitor):
        def __init__(self):
            self.depth = 0

        def visit_FunctionDef(self, node):
            # 2. Function LOC limit (exclude docstring if possible or just use ast lines)
            start = node.lineno
            # ast node has end_lineno in Python 3.8+
            end = getattr(node, "end_lineno", start)
            func_loc = end - start + 1
            if func_loc > 60:
                warnings.append(
                    f"WARNING: Function '{node.name}' at line {start} exceeds 60 LOC ({func_loc} lines)"
                )

            # Reset depth for functions
            old_depth = self.depth
            self.depth = 0
            self.generic_visit(node)
            self.depth = old_depth

        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)

        def visit_If(self, node):
            self.depth += 1
            if self.depth > 3:
                warnings.append(f"WARNING: Nesting depth {self.depth} at line {node.lineno} (If)")
            self.generic_visit(node)
            self.depth -= 1

        def visit_For(self, node):
            self.depth += 1
            if self.depth > 3:
                warnings.append(f"WARNING: Nesting depth {self.depth} at line {node.lineno} (For)")
            self.generic_visit(node)
            self.depth -= 1

        def visit_While(self, node):
            self.depth += 1
            if self.depth > 3:
                warnings.append(
                    f"WARNING: Nesting depth {self.depth} at line {node.lineno} (While)"
                )
            self.generic_visit(node)
            self.depth -= 1

        def visit_Try(self, node):
            self.depth += 1
            if self.depth > 3:
                warnings.append(f"WARNING: Nesting depth {self.depth} at line {node.lineno} (Try)")
            self.generic_visit(node)
            self.depth -= 1

    visitor = ComplexityVisitor()
    visitor.visit(tree)
    return warnings


def main():
    exclude_dirs = {".venv", ".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".deepeval"}
    target_dirs = ["app", "ai_service", "checkpointing", "scripts"]

    total_warnings = 0
    for target in target_dirs:
        for root, dirs, files in os.walk(target):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    warnings = lint_file(path)
                    if warnings:
                        print(f"\n{path}:")
                        for w in warnings:
                            print(f"  {w}")
                        total_warnings += len(warnings)

    print(f"\nTotal custom lint warnings: {total_warnings}")
    # Always exit with 0 to warn rather than block git commit unless we want it as a strict checker.
    # The mentor feedback says "warning", so we exit 0.
    sys.exit(0)


if __name__ == "__main__":
    main()
