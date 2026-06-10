import os
from pathlib import Path
import yaml

_DIR = Path(__file__).resolve().parent


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = os.environ.get("G3_CONFIG", str(_DIR / "config.yaml"))

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
