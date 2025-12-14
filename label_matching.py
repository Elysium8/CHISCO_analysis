import json
from typing import Dict

def load_text_to_class_map(path: str) -> Dict[str, int]:
    """Load sentence -> class_id mapping from textmaps.json."""
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)  # keys: sentence (str), values: int
    return mapping


def load_class_id_to_name_map(path: str):
    """Load class_id (int) -> category_name (str) from classnumber.json."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)  # keys are strings "0","1",..., values are category names
    id2name = {int(k): v for k, v in raw.items()}
    return id2name


def num_classes_from_classnumber(path: str) -> int:
    id2name = load_class_id_to_name_map(path)
    return len(id2name)
