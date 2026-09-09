"""Small replay-verified Molab smoke test for IHES puzzle 9.

Input: public model ``arabidopsisthalian/ihes-model-1778521793``.
Output: ``e001_result.json`` plus one compact JSON line on stdout.

This run is deliberately cheap.  It proves GPU, model loading, all 18 moves,
beam search, path decoding and full 72-position replay before larger jobs.
Failed checks are reported in JSON and never used as stop-the-world canaries.
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import time
import urllib.request
import zipfile

import numpy as np
import torch

from ihes_dual.beam import BeamConfig, beam_search
from ihes_dual.model import load_mlp2rb
from ihes_dual.puzzle import IHESPuzzle, inverse_move_name
from ihes_dual.registry import resolve_model


MODEL_ID = "1778521793"
MODEL_DATASET = "arabidopsisthalian/ihes-model-1778521793"
PUZZLE_ID = 9
BEAM_WIDTH = 2**14
MAX_DEPTH = 12
ORGANIZER_PATH = "-r0.d2.f2.-r1.-r2.r0.-d1"
START_STATE = [
    9, 3, 11, 44, 43, 42, 41, 40, 70, 38, 37, 69, 0, 1, 57, 25, 29, 31,
    28, 30, 20, 35, 32, 23, 47, 46, 14, 27, 65, 67, 64, 66, 61, 33, 63, 2,
    24, 49, 50, 22, 52, 53, 54, 55, 56, 36, 34, 59, 12, 51, 48, 15, 4, 5,
    6, 7, 8, 21, 10, 71, 45, 39, 62, 13, 17, 19, 16, 18, 58, 60, 68, 26,
]
PUZZLE_BLOB = (
    "H4sIAAAAAAAACrXYy4rcOhAG4H2eouj1H5BKUtmaVwnhMJxMsjnkQGd2Ie8eflmalj0ycV+y"
    "MA3dvpS+qpKl/vlBROT078v31/Pzf//8eH1+fTk9fXLwUAREJBgmzMjwDt7DK3yAj/AJ3uAn"
    "+Bk+Qx3UQxUaoBGaoAadoDM0IzgEj6AIASEiJARDmBBmhIzoED2iIgbEiJgQDXFCnBEzkkPy"
    "SIoUkCJSQjKkCWlGyjAH8zCFBViEJZjBJtgMy5gcJv8ZyyC/vXx/OT+//n/+cXr6efrqTk/y"
    "SSEzxHuIhyTIBIkQg3gHcZAAyZA4Q2KGJAfxCeJ5wgTxvDZD1EHUQ1QhGiBeIZ6fEaITRGeI"
    "ZkjgHT0kKCQESIiQkCDK8/hpkMDnOUj0kKiQGCAxQmKCRIPECRJ4Hj9nSGLgCkkBkiIkJUgy"
    "SJogib9niDmIeYgpxALEOMYEMYPYBLEZYhkyOcibVzH7uDAx9uJBFgZRpCix4HF862EcYeqHc"
    "YSpT8MRpj4Nf5fpq6cSfQoEYyk8VpzmUkSkK4V2IKpe6Yhqr3REtVc6ovqoYrqRaW8Ye0x7w"
    "9hj2kvDHtNeGh5UTHqb0pGoeqUjqr3SEdVe6YgqBYoOFVzVoRR/45g9daqWrovpwUx7E/KRHj"
    "rC1KfhCFOfhlJEc6XgkSoRqazS8eC5uWc6u4sSI94wMdjqxDE0Jx0zzZeoqFpajfSUSlWI31"
    "OBd6RibV/qMvqmxNGMlDjSUctdM38zy6OWY5Zby2lYFVPPxMs3TLyyMvXvd9txsg6qDJqkvA"
    "3vzMNVJJ7D8mNieDDdvFcHteP0ACaWw4iJNdGYgq2Kqc3ffHTtOR4Mhjd0re8Y/FvfXdpuS0"
    "qjZnqk5bbzd4mwtlxpB46AT4iXxUAZ6Y1KLcdUGuWYSkuO18X09ppbWmlw9cLUGolM1jmNCp"
    "dW0x1WvrMiVUlF53QHE6uhMW3roTGxItZM5zJ/l3XaUs7x3QzOeNsUXjlrc44m8Ap6KxAjLJ"
    "EO5u8ysjrfc8RFgL9TgUesUhQKVTBXLXdJcJswRouBluBtMRWm2kj18g1TayMytbJtc+AIyh4"
    "gtQPVOZHJVQKemWshpUrFQbDY2vqLxTddmMrLYMDEmnjP9MX9YTHQv4fKkrJN5aH65GrFc8l"
    "cqbWG04wY3siI7qOl5ZF9Sus55m/Uc8xt67m8Vr1yAf5Hpr5oS1nx8Ry+1hIiUa5kLB8evF9"
    "c+q/tU9xYqbzY7lQq/TdQYmaaEjPWv3SvY/qy3qbsLW2uXF9uNyul52vXbddzresY961WzGa"
    "j6v7YaBN4eL9N2WvRvWJaM+3Ng3cyhU6p++OhCc23AxG3CW33HU2JSdoy7a3gdoqpzN/9Uv"
    "nuncpo28tl5mj+LkukOn8zwLIH4O+EsApELEZVl50FiINc/tZ52/WOVwPuuj3vZb5fF1Nh6t"
    "+rf4XJjZXqGr1sWJZxl4JhQdGCETEcOtWFdym46llcKxK9R0rMzzVMl+3A518ffv0GpADsP6"
    "cVAAA="
)


def load_puzzle() -> IHESPuzzle:
    """Decode the audited official 72-state generator table."""

    payload = json.loads(gzip.decompress(base64.b64decode(PUZZLE_BLOB)))
    names = tuple(payload["generators"])
    moves = np.asarray([payload["generators"][name] for name in names], dtype=np.uint8)
    solved = np.asarray(payload["central_state"], dtype=np.uint8)
    lookup = {name: index for index, name in enumerate(names)}
    inverse = np.asarray([lookup[inverse_move_name(name)] for name in names], dtype=np.int16)
    puzzle = IHESPuzzle(solved=solved, moves=moves, move_names=names, inverse_move=inverse)
    puzzle.validate()
    return puzzle


def download_model_root() -> Path:
    """Return the public checkpoint directory without requiring Molab secrets.

    If Molab already has ``kagglehub`` it is reused.  Otherwise the same public
    Kaggle dataset is downloaded with the Python standard library into a
    persistent temporary cache.
    """

    try:
        import kagglehub

        return Path(kagglehub.dataset_download(MODEL_DATASET))
    except ModuleNotFoundError:
        cache = Path(tempfile.gettempdir()) / "ihes-model-1778521793"
        complete = cache / ".download_complete"
        if complete.exists():
            return cache
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / "dataset.zip.partial"
        url = f"https://www.kaggle.com/api/v1/datasets/download/{MODEL_DATASET}"
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(cache)
        archive.unlink(missing_ok=True)
        complete.write_text("ok\n", encoding="utf-8")
        return cache


def run_smoke() -> dict[str, object]:
    """Run one small GPU beam and return a self-contained result record."""

    started = time.time()
    result: dict[str, object] = {
        "experiment": "E001",
        "puzzle_id": PUZZLE_ID,
        "beam_width": BEAM_WIDTH,
        "max_depth": MAX_DEPTH,
        "model_id": MODEL_ID,
        "gpu_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        puzzle = load_puzzle()
        organizer_indices = puzzle.decode_path(ORGANIZER_PATH)
        result["organizer_length"] = len(organizer_indices)
        result["organizer_valid"] = puzzle.verify_solution(START_STATE, organizer_indices)
        result["generator_count"] = puzzle.generator_count
        if not torch.cuda.is_available():
            result["status"] = "no_cuda"
            return result

        model_root = download_model_root()
        model_spec = resolve_model(model_root, MODEL_ID)
        model = load_mlp2rb(model_spec, "cuda")
        config = BeamConfig(
            beam_width=BEAM_WIDTH,
            max_depth=MAX_DEPTH,
            parent_chunk=4096,
            inference_batch=8192,
            device="cuda",
            autocast=True,
            smaller_is_better=True,
            prune_immediate_inverse=True,
        )
        trace = beam_search(
            puzzle,
            START_STATE,
            model,
            config,
            diagnostic_path=organizer_indices,
        )
        solution = trace.solution
        valid = solution is not None and puzzle.verify_solution(START_STATE, solution)
        result.update(
            {
                "status": "solved" if solution is not None else "not_found",
                "solution": None if solution is None else puzzle.encode_path(solution),
                "length": None if solution is None else len(solution),
                "replay_valid": bool(valid),
                "first_organizer_path_drop": trace.first_natural_drop,
                "depths": [
                    {
                        "depth": item.depth,
                        "frontier": item.frontier_size,
                        "organizer_state_in_beam": item.known_state_natural_top_k,
                    }
                    for item in trace.diagnostics
                ],
            }
        )
    except Exception as error:
        result.update(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    finally:
        result["elapsed_seconds"] = round(time.time() - started, 3)
    return result


def main() -> None:
    """Save and print the result so Molab output is sufficient for recovery."""

    result = run_smoke()
    Path("e001_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("E001_RESULT", json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
