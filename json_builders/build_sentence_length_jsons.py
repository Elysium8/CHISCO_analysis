# make_sentence_length_json.py

import json
import re

# Paths (edit if needed)
TEXTMAPS_PATH = "../ds005170/json/textmaps.json"
OUTPUT_PATH = "../ds005170/json/sentence_length_chars.json"

# Remove Chinese + English punctuation and whitespace
PUNCT = re.compile(r"[，。！？、；：《》“”（）【】,.!?;:\"'()\s]+")

def sentence_length(sentence: str) -> int:
    return len(PUNCT.sub("", sentence))

# Load textmaps
with open(TEXTMAPS_PATH, "r", encoding="utf-8") as f:
    textmaps = json.load(f)

# Build mapping
length_map = {
    sentence: sentence_length(sentence)
    for sentence in textmaps.keys()
}

# Save
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(length_map, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(length_map)} sentences to {OUTPUT_PATH}")
