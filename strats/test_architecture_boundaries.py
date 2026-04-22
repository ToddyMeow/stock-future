from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _iter_python_files(base: Path):
    for path in base.rglob("*.py"):
        if path.name.startswith("test_"):
            continue
        if "__pycache__" in path.parts:
            continue
        if "_archive" in path.parts:
            continue
        yield path


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    return imported


def test_live_layer_does_not_import_scripts():
    live_dir = ROOT / "live"
    offenders = []
    for path in _iter_python_files(live_dir):
        for module in _imports(path):
            if module == "scripts" or module.startswith("scripts."):
                offenders.append((path.relative_to(ROOT), module))
    assert offenders == []


def test_strats_layer_does_not_import_scripts():
    strats_dir = ROOT / "strats"
    offenders = []
    for path in _iter_python_files(strats_dir):
        for module in _imports(path):
            if module == "scripts" or module.startswith("scripts."):
                offenders.append((path.relative_to(ROOT), module))
    assert offenders == []


def test_strategy_engine_constructor_uses_strategy_slots_only():
    offenders = []
    for base in (ROOT / "strats", ROOT / "live", ROOT / "scripts"):
        for path in _iter_python_files(base):
            tree = ast.parse(
                path.read_text(encoding="utf-8", errors="ignore"),
                filename=str(path),
            )
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                is_strategy_engine = (
                    isinstance(func, ast.Name) and func.id == "StrategyEngine"
                ) or (
                    isinstance(func, ast.Attribute) and func.attr == "StrategyEngine"
                )
                if not is_strategy_engine:
                    continue
                keywords = {kw.arg for kw in node.keywords if kw.arg is not None}
                if "entry_strategy" in keywords or "exit_strategy" in keywords:
                    offenders.append(
                        (path.relative_to(ROOT), node.lineno, sorted(keywords))
                    )
    assert offenders == []
