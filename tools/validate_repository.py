from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NOTEBOOKS = (
    "01_base_mlp_beam.ipynb",
    "02_symmetry_reverse.ipynb",
    "03_bidirectional_symmetry_reverse.ipynb",
)
CYRILLIC = re.compile(r"[\u0400-\u04ff]")
SECRET_PATTERNS = (
    re.compile(r"kaggle\.json", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|token|password)\s*[=:]\s*['\"][^'\"]+", re.IGNORECASE),
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"ghp_[0-9A-Za-z]{20,}"),
)


def validate_notebook(path: Path) -> None:
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if payload.get("nbformat") != 4 or not isinstance(payload.get("cells"), list):
        raise AssertionError(f"invalid notebook structure: {path}")
    if CYRILLIC.search(raw):
        raise AssertionError(f"Kaggle notebook contains Cyrillic text: {path}")
    for pattern in SECRET_PATTERNS:
        if pattern.search(raw):
            raise AssertionError(f"possible secret in {path}: {pattern.pattern}")
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        try:
            ast.parse(source, filename=f"{path}:cell-{index}")
        except SyntaxError as error:
            raise AssertionError(f"invalid Python in {path} cell {index}: {error}") from error


def main() -> None:
    for name in EXPECTED_NOTEBOOKS:
        validate_notebook(ROOT / "notebooks" / name)
    forbidden_suffixes = {".pth", ".pt", ".rar", ".7z"}
    for path in ROOT.rglob("*"):
        if any(part in {".git", "work"} for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in forbidden_suffixes:
            raise AssertionError(f"heavy/private asset is tracked in the repository tree: {path}")
        if path.is_file() and path.stat().st_size > 10 * 1024 * 1024:
            raise AssertionError(f"unexpected file larger than 10 MiB: {path}")
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    if any(Path(name).name.lower() == "kaggle.json" for name in tracked):
        raise AssertionError("kaggle.json must never be tracked")
    print("repository validation passed")


if __name__ == "__main__":
    main()
