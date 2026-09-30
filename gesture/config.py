from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path=ROOT / "config.yaml"):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
