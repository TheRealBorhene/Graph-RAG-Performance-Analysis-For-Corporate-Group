import json
from pathlib import Path


def load_document(file_path: str) -> str:
    """Load text from a .md or .json file."""
    path = Path(file_path)

    if path.suffix == ".md":
        return path.read_text(encoding="utf-8")

    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return "\n".join(f"{k}: {v}" for k, v in data.items())

    raise ValueError(f"Unsupported file type: {path.suffix}")
