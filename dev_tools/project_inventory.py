from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "hakyking"
SKIP_DIRS = {".venv", "__pycache__", ".git"}


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if path.is_dir():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def module_summary(path: Path) -> tuple[list[str], list[str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - inventory should not fail hard
        return [], [f"parse_error={exc}"]

    classes: list[str] = []
    functions: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
    return classes, functions


def main() -> None:
    print("# Hakyking Project Inventory")
    print(f"root={ROOT}")
    print(f"package={PACKAGE}")
    print()

    files = iter_files()
    print("## Top-level Files")
    for path in sorted(ROOT.iterdir()):
        if path.name in SKIP_DIRS:
            continue
        rel = path.relative_to(ROOT)
        kind = "dir" if path.is_dir() else "file"
        print(f"- {kind}: {rel}")
    print()

    print("## Python Modules")
    for path in files:
        if path.suffix != ".py" or not path.is_relative_to(PACKAGE):
            continue
        rel = path.relative_to(ROOT)
        classes, functions = module_summary(path)
        parts: list[str] = []
        if classes:
            parts.append("classes=" + ",".join(classes))
        if functions:
            parts.append("functions=" + ",".join(functions))
        suffix = " | " + " | ".join(parts) if parts else ""
        print(f"- {rel}{suffix}")
    print()

    print("## Non-Python Project Files")
    for path in files:
        if path.suffix == ".py":
            continue
        rel = path.relative_to(ROOT)
        print(f"- {rel}")
    print()

    print("## Suggested Recovery Reads")
    for rel in [
        "AI_HANDOFF.md",
        "ARCHITECTURE.md",
        "README.md",
        "hakyking/views/main_window.py",
        "hakyking/controllers/main_controller.py",
        "hakyking/views/workspace.py",
        "hakyking/views/workspace_boundary.py",
        "hakyking/audio/audio_engine.py",
        "hakyking/project_manager.py",
    ]:
        print(f"- {ROOT / rel}")


if __name__ == "__main__":
    main()
