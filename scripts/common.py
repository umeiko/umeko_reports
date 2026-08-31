from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["_config_path"] = str(config_path)
    return payload


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def environment_snapshot() -> dict[str, Any]:
    try:
        import torch

        torch_info = {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        }
    except Exception as error:  # pragma: no cover - diagnostic path
        torch_info = {"error": repr(error)}
    try:
        uv_version = subprocess.check_output(["uv", "--version"], text=True).strip()
    except Exception:
        uv_version = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "uv": uv_version,
        **torch_info,
    }
