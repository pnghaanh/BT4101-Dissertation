import json
from pathlib import Path
from typing import Dict


def save_summary_json(path: str, payload: Dict):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
