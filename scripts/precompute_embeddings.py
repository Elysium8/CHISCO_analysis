"""Compute the sentence-embedding targets: the [CLS] vector of bert-base-chinese for every CHISCO sentence.

Reads the sentences from the dataset's textmaps.json and writes {sentence: [768 floats]}.

    python -m scripts.precompute_embeddings
"""

import argparse
import json

import torch
from transformers import AutoModel, AutoTokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--textmaps", default="./ds005170/json/textmaps.json")
    ap.add_argument("--out", default="./ds005170/json/sentence_bert_embs.json")
    ap.add_argument("--model", default="bert-base-chinese")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval()

    with open(args.textmaps, "r", encoding="utf-8") as f:
        sentences = list(json.load(f))

    embeddings = {}
    with torch.no_grad():
        for s in sentences:
            output = model(**tokenizer(s, return_tensors="pt", truncation=True))
            embeddings[s] = output.last_hidden_state[0, 0].tolist()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False)
    print(f"Wrote {len(embeddings)} embeddings to {args.out}")


if __name__ == "__main__":
    main()
