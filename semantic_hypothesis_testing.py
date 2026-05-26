#!/usr/bin/env python3
import os
import sys
import json
import argparse
import math
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.append("./ds005170")
from data_loader import load_read_run, load_imagine_run  # type: ignore

# IMPORTANT: use the model class from your new training script
from train_script_with_split import EEGToTextEmbedModel


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sentence_embeddings(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_text_embedding_matrix(
    sent2emb: Dict[str, List[float]]
) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
    sentences = sorted(sent2emb.keys())
    sentence_to_id = {s: i for i, s in enumerate(sentences)}
    matrix = np.stack(
        [np.asarray(sent2emb[s], dtype=np.float32) for s in sentences],
        axis=0,
    )
    matrix = torch.tensor(matrix)
    matrix = torch.nn.functional.normalize(matrix, dim=1)
    return matrix, sentences, sentence_to_id


def normalize_sentence(s):
    return str(s).strip()


def indices_from_sentence_split(sentences, split_sentences):
    split_set = set(normalize_sentence(s) for s in split_sentences)
    return [
        i for i, s in enumerate(sentences)
        if normalize_sentence(s) in split_set
    ]


def discover_experiment_dirs(root: str) -> List[str]:
    run_dirs = []
    for dirpath, _, filenames in os.walk(root):
        files = set(filenames)
        if {"config.json", "split.json", "best_checkpoint.pt"}.issubset(files):
            run_dirs.append(dirpath)
    return sorted(run_dirs)


def mean_ci95(values: List[float]) -> Tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, 0.0
    sem = float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return mean, 1.96 * sem


def normalize_task_group(task: str) -> str:
    task = str(task).strip().lower()
    if task == "read":
        return "read"
    if task in {"imagine", "recall"}:
        return "recall"
    return task


@dataclass(frozen=True)
class TrialRef:
    sub_code: str
    run_idx: int
    trial_in_run: int
    subject_index: int
    sentence: str
    task: str


class ChiscoStreamingEvalDataset(Dataset):
    def __init__(self, root: str, trial_index: List[TrialRef]):
        self.root = root
        self.trial_index = trial_index
        self._cache_key: Optional[Tuple[str, int, str]] = None
        self._cache_trials: Optional[List[dict]] = None

    def __len__(self) -> int:
        return len(self.trial_index)

    def _load_run_trials(self, sub_code: str, run_idx: int, task: str) -> List[dict]:
        key = (sub_code, run_idx, task)

        if self._cache_key == key and self._cache_trials is not None:
            return self._cache_trials

        if task == "read":
            trials = load_read_run(self.root, sub_code, run_idx)
        elif task == "imagine":
            trials = load_imagine_run(self.root, sub_code, run_idx)
        else:
            raise ValueError(f"Unknown task: {task}")

        self._cache_key = key
        self._cache_trials = trials
        return trials

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ref = self.trial_index[idx]
        trials = self._load_run_trials(ref.sub_code, ref.run_idx, ref.task)
        trial = trials[ref.trial_in_run]

        arr = trial["input_features"]
        eeg = arr[0, :122, :] * 1_000_000.0
        eeg = np.asarray(eeg, dtype=np.float32)
        eeg = torch.from_numpy(eeg)

        return {
            "eeg": eeg,
            "subject_id": torch.tensor(ref.subject_index, dtype=torch.long),
            "sentence": ref.sentence,
        }


def build_streaming_index(
    root: str,
    subject_codes: List[str],
    task: str,
    n_runs_read: int = 45,
    n_runs_imagine: int = 45,
) -> Tuple[List[TrialRef], List[str], torch.Tensor]:
    trial_index: List[TrialRef] = []
    sentences: List[str] = []
    subj_ids: List[int] = []

    if task == "read":
        n_runs = n_runs_read
        run_loader = load_read_run
    elif task == "imagine":
        n_runs = n_runs_imagine
        run_loader = load_imagine_run
    else:
        raise ValueError(f"Unknown task: {task}")

    for subj_idx, sub_code in enumerate(subject_codes):
        for run_idx in range(1, n_runs + 1):
            trials = run_loader(root, sub_code, run_idx)

            if not trials:
                continue

            for t_i, trial in enumerate(trials):
                sent = str(trial.get("text") or "").strip()

                if not sent:
                    continue

                trial_index.append(
                    TrialRef(
                        sub_code=sub_code,
                        run_idx=run_idx,
                        trial_in_run=t_i,
                        subject_index=subj_idx,
                        sentence=sent,
                        task=task,
                    )
                )
                sentences.append(sent)
                subj_ids.append(subj_idx)

    return trial_index, sentences, torch.tensor(subj_ids, dtype=torch.long)


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _coerce_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def build_semantic_index(
    text_sentences: List[str],
    textmaps_path: str,
    classnumber_path: str,
) -> Tuple[Dict[str, int], Dict[int, str], Dict[int, int], int]:
    sent2class_raw: Dict[str, Any] = read_json(textmaps_path)
    class_id_to_name_raw: Dict[str, str] = read_json(classnumber_path)

    class_id_to_name: Dict[int, str] = {}
    for k, v in class_id_to_name_raw.items():
        kid = _coerce_int(k)
        if kid is not None:
            class_id_to_name[kid] = str(v)

    sentence_to_class: Dict[str, int] = {}
    class_counts: Dict[int, int] = {}

    for s in text_sentences:
        if s not in sent2class_raw:
            continue

        cid = _coerce_int(sent2class_raw[s])
        if cid is None:
            continue

        sentence_to_class[s] = cid
        class_counts[cid] = class_counts.get(cid, 0) + 1

    N = sum(class_counts.values())
    return sentence_to_class, class_id_to_name, class_counts, N


@torch.no_grad()
def evaluate_global_retrieval_streaming_with_semantics(
    model,
    dataset: Dataset,
    text_matrix: torch.Tensor,
    text_id_to_sentence: List[str],
    sentence_to_id: Dict[str, int],
    sentence_to_class: Dict[str, int],
    class_counts: Dict[int, int],
    N_classed: int,
    device: torch.device,
    batch_size: int = 32,
    semantic_topn: int = 10,
    semantic_topk_test: int = 50,
    include_target_in_semantic: bool = False,
) -> Dict[str, Any]:
    model.eval()
    text_matrix = text_matrix.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    ks = [1, 5, 10, 50]
    top_counts = {k: 0 for k in ks}
    total = 0

    n_use_topn = int(max(1, semantic_topn))
    n_use_topk = int(max(1, semantic_topk_test))

    sem_total_topn = 0
    sem_hits_topn = 0

    sem_total_topk = 0
    sem_hits_topk = 0

    null_mean_sum = 0.0
    null_var_sum = 0.0

    for batch in loader:
        eeg = batch["eeg"].to(device)
        subj = batch["subject_id"].to(device)
        batch_sents: List[str] = list(batch["sentence"])

        eeg_emb = model(eeg, subj)
        eeg_emb = torch.nn.functional.normalize(eeg_emb, dim=1)

        sim = eeg_emb @ text_matrix.T
        k_use = min(max(50, n_use_topk, n_use_topn), sim.shape[1])
        _, indices = sim.topk(k=k_use, dim=1)

        for i, target_sent in enumerate(batch_sents):
            total += 1

            true_id = sentence_to_id.get(target_sent, None)
            if true_id is None:
                continue

            cand_ids = indices[i].tolist()

            if true_id == cand_ids[0]:
                top_counts[1] += 1
            if true_id in cand_ids[:5]:
                top_counts[5] += 1
            if true_id in cand_ids[:10]:
                top_counts[10] += 1
            if true_id in cand_ids[:50]:
                top_counts[50] += 1

            target_cls = sentence_to_class.get(target_sent, None)
            if target_cls is None:
                continue

            cand_sents = [text_id_to_sentence[j] for j in cand_ids]

            def is_semantic_hit(s: str) -> bool:
                if (not include_target_in_semantic) and (s == target_sent):
                    return False
                return sentence_to_class.get(s, None) == target_cls

            topn_slice = cand_sents[:n_use_topn]
            sem_hits_topn += sum(1 for s in topn_slice if is_semantic_hit(s))
            sem_total_topn += len(topn_slice)

            topk_slice = cand_sents[:n_use_topk]
            x = sum(1 for s in topk_slice if is_semantic_hit(s))

            sem_hits_topk += x
            sem_total_topk += len(topk_slice)

            K = class_counts.get(target_cls, 0)

            if N_classed > 1 and 0 <= K <= N_classed:
                p = K / float(N_classed)
                n = float(len(topk_slice))
                mu = n * p
                var = n * p * (1.0 - p) * ((N_classed - n) / (N_classed - 1.0))

                null_mean_sum += mu
                null_var_sum += var

    sem_frac_topn = sem_hits_topn / sem_total_topn if sem_total_topn else float("nan")
    sem_frac_topk = sem_hits_topk / sem_total_topk if sem_total_topk else float("nan")

    expected_topk = null_mean_sum
    enrichment = sem_hits_topk / expected_topk if expected_topk > 0 else float("inf")

    if null_var_sum > 0:
        z = (sem_hits_topk - expected_topk) / math.sqrt(null_var_sum)
        p_value = _norm_sf(z)
    else:
        z = float("nan")
        p_value = float("nan")

    return {
        "top1": top_counts[1] / total * 100.0,
        "top5": top_counts[5] / total * 100.0,
        "top10": top_counts[10] / total * 100.0,
        "top50": top_counts[50] / total * 100.0,
        "total": float(total),
        "semantic": {
            "semantic_topn": int(n_use_topn),
            "semantic_topk_test": int(n_use_topk),
            "include_target_in_semantic": bool(include_target_in_semantic),
            "topn_same_class_fraction": float(sem_frac_topn),
            "topk_same_class_fraction": float(sem_frac_topk),
            "topk_same_class_count": int(sem_hits_topk),
            "topk_total_count": int(sem_total_topk),
            "topk_expected_same_class_count": float(expected_topk),
            "topk_enrichment_vs_chance": float(enrichment),
            "z": float(z),
            "p_value_one_sided": float(p_value),
        },
    }


def build_model_kwargs(
    model_cls,
    cfg_dict: Dict[str, Any],
    subject_codes: List[str],
    text_dim: int,
) -> Dict[str, Any]:
    sig = inspect.signature(model_cls.__init__)
    params = set(sig.parameters.keys())

    n_channels = int(cfg_dict.get("n_channels", cfg_dict.get("eeg_channels", 122)))
    n_subjects = int(cfg_dict.get("n_subjects", len(subject_codes)))
    pooling = str(cfg_dict.get("pooling", "mean"))

    use_frontend = bool(cfg_dict.get("use_frontend", True))
    use_brainmodule = bool(cfg_dict.get("use_brainmodule", True))

    use_eegnet = bool(cfg_dict.get("use_eegnet", True))
    eegnet_temporal_kern = int(cfg_dict.get("eegnet_temporal_kern", 64))
    eegnet_F1 = int(cfg_dict.get("eegnet_F1", cfg_dict.get("eegnet_f1", 16)))
    eegnet_D = int(cfg_dict.get("eegnet_D", cfg_dict.get("eegnet_d", 2)))

    use_spatial_attn = bool(cfg_dict.get("use_spatial_attn", True))
    use_subject_layer = bool(cfg_dict.get("use_subject_layer", True))
    use_residual_blocks = bool(cfg_dict.get("use_residual_blocks", True))

    raw_frontend_out_ch = int(cfg_dict.get("raw_frontend_out_ch", 32))
    n_blocks = int(cfg_dict.get("n_blocks", 5))

    if "n_channels" in params and "text_dim" in params:
        return dict(
            n_channels=n_channels,
            n_subjects=n_subjects,
            text_dim=text_dim,
            latent_dim=int(cfg_dict.get("latent_dim", 256)),
            hidden_channels=int(cfg_dict.get("hidden_channels", 256)),
            final_hidden=int(cfg_dict.get("final_hidden", 256)),
            n_blocks=n_blocks,
            pooling=pooling,
            use_frontend=use_frontend,
            use_brainmodule=use_brainmodule,
            use_eegnet=use_eegnet,
            eegnet_temporal_kern=eegnet_temporal_kern,
            eegnet_F1=eegnet_F1,
            eegnet_D=eegnet_D,
            raw_frontend_out_ch=raw_frontend_out_ch,
            use_spatial_attn=use_spatial_attn,
            use_subject_layer=use_subject_layer,
            use_residual_blocks=use_residual_blocks,
        )

    raise TypeError(
        "Unsupported EEGToTextEmbedModel constructor. Detected parameters: "
        + ", ".join(sorted(params))
    )


def evaluate_one_experiment(
    args: argparse.Namespace,
    exp_dir: str,
    device: torch.device,
) -> Dict[str, Any]:
    cfg = read_json(os.path.join(exp_dir, "config.json"))
    split_info = read_json(os.path.join(exp_dir, "split.json"))

    ckpt_path = args.ckpt or os.path.join(exp_dir, "best_checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    chisco_root = cfg.get("chisco_root") or cfg.get("CHISCO_ROOT") or "./ds005170/derivatives/preprocessed_pkl"
    sent_emb_path = cfg.get("sent_emb_path") or cfg.get("SENT_EMB_PATH") or "./ds005170/json/sentence_bert_embs.json"
    task = str(cfg.get("task") or cfg.get("TASK") or "read").strip().lower()

    subject_codes_str = cfg.get("subject_codes")
    if isinstance(subject_codes_str, str):
        subject_codes = [s.strip() for s in subject_codes_str.split(",") if s.strip()]
    elif isinstance(subject_codes_str, list):
        subject_codes = [str(s).strip() for s in subject_codes_str if str(s).strip()]
    else:
        subject_codes = ["01", "02", "03", "04", "05"]

    trial_index, trial_sentences, subj_ids = build_streaming_index(
        root=chisco_root,
        subject_codes=subject_codes,
        task=task,
    )
    _ = subj_ids

    if "test_sentences" in split_info:
        eval_idx = indices_from_sentence_split(
            trial_sentences,
            split_info["test_sentences"],
        )
        eval_split_name = "test"
    elif "test_idx" in split_info:
        eval_idx = split_info["test_idx"]
        eval_split_name = "test"
    elif "val_sentences" in split_info:
        eval_idx = indices_from_sentence_split(
            trial_sentences,
            split_info["val_sentences"],
        )
        eval_split_name = "val"
    elif "val_idx" in split_info:
        eval_idx = split_info["val_idx"]
        eval_split_name = "val"
    else:
        raise KeyError("split.json must contain test_sentences/test_idx or val_sentences/val_idx")

    if not eval_idx:
        raise ValueError(f"{eval_split_name} split produced no samples.")

    sent2emb = load_sentence_embeddings(sent_emb_path)
    text_matrix, text_id_to_sentence, sentence_to_id = build_text_embedding_matrix(sent2emb)
    text_dim = int(text_matrix.shape[1])

    sentence_to_class, class_id_to_name, class_counts, N_classed = build_semantic_index(
        text_sentences=text_id_to_sentence,
        textmaps_path=args.textmaps,
        classnumber_path=args.classnumber,
    )
    _ = class_id_to_name

    print(
        f"[{os.path.basename(exp_dir)}] task={task} | split={eval_split_name} | "
        f"trials={len(eval_idx)} | semantic coverage={N_classed}/{len(text_id_to_sentence)}"
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_cfg = checkpoint.get("config", cfg)

    model_kwargs = build_model_kwargs(
        EEGToTextEmbedModel,
        model_cfg,
        subject_codes=subject_codes,
        text_dim=text_dim,
    )

    model = EEGToTextEmbedModel(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    base_ds = ChiscoStreamingEvalDataset(root=chisco_root, trial_index=trial_index)
    eval_ds = Subset(base_ds, eval_idx)

    results = evaluate_global_retrieval_streaming_with_semantics(
        model=model,
        dataset=eval_ds,
        text_matrix=text_matrix,
        text_id_to_sentence=text_id_to_sentence,
        sentence_to_id=sentence_to_id,
        sentence_to_class=sentence_to_class,
        class_counts=class_counts,
        N_classed=N_classed,
        device=device,
        batch_size=args.eval_batch_size,
        semantic_topn=args.semantic_topn,
        semantic_topk_test=args.semantic_topk,
        include_target_in_semantic=args.include_target_in_semantic,
    )

    results["exp_dir"] = exp_dir
    results["task"] = task
    results["task_group"] = normalize_task_group(task)
    results["split_mode"] = split_info.get("split_mode")
    results["eval_split"] = eval_split_name

    out_path = os.path.join(exp_dir, f"global_retrieval_semantic_{eval_split_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved per-run semantic results: {out_path}")
    return results


def summarize_group(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_values = {
        "top1": [r["top1"] for r in results],
        "top5": [r["top5"] for r in results],
        "top10": [r["top10"] for r in results],
        "top50": [r["top50"] for r in results],
        "same_class_topn_fraction_percent": [
            100.0 * r["semantic"]["topn_same_class_fraction"] for r in results
        ],
        "same_class_topk_fraction_percent": [
            100.0 * r["semantic"]["topk_same_class_fraction"] for r in results
        ],
        "enrichment_vs_chance": [
            r["semantic"]["topk_enrichment_vs_chance"] for r in results
        ],
        "z": [
            r["semantic"]["z"] for r in results
        ],
        "p_value_one_sided": [
            r["semantic"]["p_value_one_sided"] for r in results
        ],
    }

    summary = {
        "n_runs": len(results),
        "run_dirs": [r["exp_dir"] for r in results],
        "metrics": {},
    }

    for name, vals in metric_values.items():
        mean, ci = mean_ci95(vals)
        summary["metrics"][name] = {
            "mean": mean,
            "ci95_pm": ci,
            "values": vals,
        }

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate global retrieval + semantic enrichment."
    )

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--exp-dir", type=str, help="Single experiment run dir")
    g.add_argument("--exp-root", type=str, help="Folder containing many experiment run dirs")

    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--eval-batch-size", type=int, default=32)

    ap.add_argument("--textmaps", type=str, default="textmaps.json")
    ap.add_argument("--classnumber", type=str, default="classnumber.json")
    ap.add_argument("--semantic-topn", type=int, default=10)
    ap.add_argument("--semantic-topk", type=int, default=50)
    ap.add_argument("--include-target-in-semantic", action="store_true")

    args = ap.parse_args()

    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    print("Using device:", device)

    if args.exp_dir:
        result = evaluate_one_experiment(args, args.exp_dir, device)

        print("\n=== Single-run summary ===")
        print(f"Task: {result['task']} | Eval split: {result['eval_split']}")
        print(f"Top-1 : {result['top1']:.2f}%")
        print(f"Top-5 : {result['top5']:.2f}%")
        print(f"Top-10: {result['top10']:.2f}%")
        print(f"Top-50: {result['top50']:.2f}%")

        sem = result["semantic"]
        print(f"Same-class top-{sem['semantic_topn']}: {100.0 * sem['topn_same_class_fraction']:.2f}%")
        print(f"Same-class top-{sem['semantic_topk_test']}: {100.0 * sem['topk_same_class_fraction']:.2f}%")
        print(f"Enrichment: {sem['topk_enrichment_vs_chance']:.2f}x")
        print(f"p-value: {sem['p_value_one_sided']:.3g}")
        return

    exp_dirs = discover_experiment_dirs(args.exp_root)
    print(f"Found {len(exp_dirs)} experiment folders")

    grouped = defaultdict(list)
    failures = []

    for exp_dir in exp_dirs:
        print(f"\nEvaluating: {exp_dir}")
        try:
            result = evaluate_one_experiment(args, exp_dir, device)
            grouped[result["task_group"]].append(result)
        except Exception as e:
            failures.append((exp_dir, repr(e)))
            print(f"FAILED: {exp_dir}")
            print(repr(e))

    summary = {
        "exp_root": args.exp_root,
        "semantic_topn": args.semantic_topn,
        "semantic_topk": args.semantic_topk,
        "groups": {},
        "failures": failures,
    }

    for group_name in ["read", "recall"]:
        runs = grouped.get(group_name, [])
        if not runs:
            continue
        summary["groups"][group_name] = summarize_group(runs)

    out_dir = args.out_dir or os.path.join(args.exp_root, "semantic_aggregate")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "semantic_aggregate_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved aggregate summary: {out_path}")

    for group_name, block in summary["groups"].items():
        print(f"\n=== {group_name.upper()} ({block['n_runs']} runs) ===")
        for metric, stats in block["metrics"].items():
            print(f"{metric}: {stats['mean']:.3f} ± {stats['ci95_pm']:.3f}")

    if failures:
        print("\nFailures:")
        for exp_dir, err in failures:
            print(f"- {exp_dir}: {err}")


if __name__ == "__main__":
    main()