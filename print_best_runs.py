import os
import json
import argparse
import subprocess
import sys
import shutil

# --------------------------------------------------
# GLOBAL TOGGLES (must BOTH be set to delete)
# --------------------------------------------------
DELETE_SUGGESTED = False
DELETE_CONFIRM_TOKEN = ""  # must be exactly "YES_DELETE" to actually delete
# --------------------------------------------------

TASKS = ["read", "imagine"]
SPLITS = ["iid", "unseen_sentence", "unseen_subject", "sentence_aligned_read_imagine"]

# Per-K metrics we care about
BASE_METRICS = ["top1", "top5", "top10", "top50"]
# Balanced "total" metric
BALANCED_METRIC = "avg"
# Metrics to report "best model" for
REPORT_METRICS = BASE_METRICS + [BALANCED_METRIC]

CORE_FILES = ["config.json", "split.json", "best_checkpoint.pt"]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_run_dirs(experiments_root: str):
    run_dirs = []
    for dirpath, _, filenames in os.walk(experiments_root):
        files = set(filenames)
        if all(x in files for x in CORE_FILES):
            run_dirs.append(dirpath)
    return run_dirs


def normalize_task(x):
    if x is None:
        return None
    x = str(x).strip().lower()
    if x in ("read", "reading"):
        return "read"
    if x in ("imagine", "imagined", "imaginary", "imagination", "imaginer"):
        return "imagine"
    return None


def infer_task_and_split(run_dir: str):
    try:
        cfg = load_json(os.path.join(run_dir, "config.json"))
    except Exception:
        cfg = {}

    try:
        split = load_json(os.path.join(run_dir, "split.json"))
    except Exception:
        split = {}

    task = normalize_task(cfg.get("task") or cfg.get("TASK"))
    split_mode = split.get("split_mode") or split.get("SPLIT_MODE")

    low = run_dir.lower()
    if task is None:
        if "/read/" in low or "\\read\\" in low:
            task = "read"
        elif "/imagine/" in low or "\\imagine\\" in low:
            task = "imagine"
        elif "/imaginary/" in low or "\\imaginary\\" in low:
            task = "imagine"
        elif "/imaginer/" in low or "\\imaginer\\" in low:
            task = "imagine"

    if split_mode not in SPLITS:
        for s in SPLITS:
            if f"/{s}/" in low or f"\\{s}\\" in low:
                split_mode = s
                break

    return task, split_mode


def run_global_eval(eval_script: str, run_dir: str, device: str, eval_batch_size: int):
    cmd = [
        sys.executable,
        eval_script,
        "--exp-dir", run_dir,
        "--device", device,
        "--eval-batch-size", str(eval_batch_size),
    ]
    subprocess.run(cmd, check=True)




def has_trial_level_results(run_dir: str) -> bool:
    """
    Returns True if a run already has trial-level global retrieval output.

    Preferred new format:
      - global_retrieval_trials.json: a list of per-trial rows

    Also accepts a transitional/legacy format where trial_rows were saved inside
    global_retrieval.json.
    """
    trial_json = os.path.join(run_dir, "global_retrieval_trials.json")
    if os.path.exists(trial_json):
        try:
            rows = load_json(trial_json)
            if not isinstance(rows, list) or len(rows) == 0:
                return False
            required = {"top1_correct", "top5_correct", "top10_correct", "top50_correct"}
            return required.issubset(set(rows[0].keys()))
        except Exception:
            return False

    aggregate_json = os.path.join(run_dir, "global_retrieval.json")
    if os.path.exists(aggregate_json):
        try:
            metrics = load_json(aggregate_json)
            rows = metrics.get("trial_rows", None)
            if not isinstance(rows, list) or len(rows) == 0:
                return False
            required = {"top1_correct", "top5_correct", "top10_correct", "top50_correct"}
            return required.issubset(set(rows[0].keys()))
        except Exception:
            return False

    return False


def fmt_metrics(mdict):
    def g(k):
        v = mdict.get(k, None)
        return "--" if v is None else f"{float(v):.2f}%"

    parts = [f"{m}={g(m)}" for m in BASE_METRICS]
    parts.append(f"{BALANCED_METRIC}={g(BALANCED_METRIC)}")
    return ", ".join(parts)


def compute_balanced_metric(mvec):
    # Simple, balanced total: macro-average over the base K accuracies
    vals = [float(mvec.get(k, 0.0)) for k in BASE_METRICS]
    return sum(vals) / max(1, len(vals))


def dominates(a_vec, b_vec):
    """
    True if a_vec dominates b_vec on BASE_METRICS:
      a >= b for all base metrics, and a > b for at least one.
    Metrics are accuracies -> higher is better.
    """
    ge_all = True
    gt_any = False
    for k in BASE_METRICS:
        if a_vec[k] < b_vec[k]:
            ge_all = False
            break
        if a_vec[k] > b_vec[k]:
            gt_any = True
    return ge_all and gt_any


def main():
    ap = argparse.ArgumentParser(
        description="Find best experiment folders per task/split using eval_global_retrieval.py, and suggest deletions. Recomputes runs that lack trial-level global retrieval output."
    )
    ap.add_argument("--experiments-root", type=str, default="Proper_split_results")
    ap.add_argument("--eval-script", type=str, default="eval_global_retrieval.py")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument(
        "--require-trial-level",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, recompute runs whose aggregate global_retrieval.json exists "
            "but trial-level output is missing or invalid. Use --no-require-trial-level "
            "to keep the old aggregate-only behavior."
        ),
    )
    ap.add_argument(
        "--suggest-deletions",
        action="store_true",
        help="If set, prints run folders that are dominated by the best metrics in their (task,split) bucket."
    )
    args = ap.parse_args()

    run_dirs = find_run_dirs(args.experiments_root)
    print(f"Found {len(run_dirs)} candidate run folders under: {args.experiments_root}\n")

    # Store per-run evaluated metrics
    # runs[(task, split)] = list of {run_dir, metrics{top1,top5,top10,top50,avg}}
    runs = {(t, s): [] for t in TASKS for s in SPLITS}

    # best[task][split][metric] = (best_value, run_dir)
    best = {
        task: {split: {m: (-1.0, None) for m in REPORT_METRICS} for split in SPLITS}
        for task in TASKS
    }

    counts = {(t, s): 0 for t in TASKS for s in SPLITS}

    for run_dir in run_dirs:
        task, split_mode = infer_task_and_split(run_dir)
        if task not in TASKS or split_mode not in SPLITS:
            continue

        counts[(task, split_mode)] += 1

        out_json = os.path.join(run_dir, "global_retrieval.json")
        needs_eval = False
        eval_reasons = []

        if args.recompute:
            needs_eval = True
            eval_reasons.append("--recompute")

        if not os.path.exists(out_json):
            needs_eval = True
            eval_reasons.append("missing global_retrieval.json")

        if args.require_trial_level and not has_trial_level_results(run_dir):
            needs_eval = True
            eval_reasons.append("missing/invalid trial-level retrieval output")

        if needs_eval:
            try:
                reason_str = "; ".join(eval_reasons)
                print(f"[EVAL] {task}/{split_mode} -> {os.path.basename(run_dir)} ({reason_str})")
                run_global_eval(args.eval_script, run_dir, args.device, args.eval_batch_size)
            except subprocess.CalledProcessError:
                print(f"  !! Eval failed for {run_dir}. Skipping.")
                continue

            if args.require_trial_level and not has_trial_level_results(run_dir):
                print(
                    "  !! Eval finished, but trial-level output was still not found. "
                    "Make sure --eval-script points to the trial-level evaluator. Skipping."
                )
                continue

        try:
            metrics = load_json(out_json)
        except Exception:
            print(f"  !! Could not read {out_json}. Skipping.")
            continue

        # Ensure we have required metrics (default missing to 0.0 so comparisons still work)
        mvec = {k: float(metrics.get(k, 0.0)) for k in BASE_METRICS}
        mvec[BALANCED_METRIC] = compute_balanced_metric(mvec)

        runs[(task, split_mode)].append({"run_dir": run_dir, "metrics": mvec})

        # Update winners for each metric independently (including avg)
        for m in REPORT_METRICS:
            val = float(mvec.get(m, 0.0))
            cur_best, _ = best[task][split_mode][m]
            if val > cur_best:
                best[task][split_mode][m] = (val, run_dir)

    print("\nDetected candidate runs per bucket:")
    for task in TASKS:
        for split in SPLITS:
            print(f"  {task:7s} / {split:14s}: {counts[(task, split)]}")
    print()

    # Print best per metric (incl avg)
    for task in TASKS:
        print("=" * 72)
        print(f"TASK: {task.upper()}")
        print("=" * 72)
        for split in SPLITS:
            print(f"\nSplit: {split.replace('_', ' ').title()}")
            for m in REPORT_METRICS:
                val, run_dir = best[task][split][m]
                if m == BALANCED_METRIC:
                    label = "Best AVG:"
                else:
                    label = f"Best {m.replace('top', 'top ').upper()}:"
                if run_dir is None:
                    print(f'  {label:<14} --')
                else:
                    print(f'  {label:<14} "{os.path.basename(run_dir)}" ({val:.2f}%)')
        print()

    # Suggest deletions
    if True:
        print("\n" + "=" * 72)
        print("SUGGESTED MODEL FOLDERS TO DELETE (DOMINATED RUNS)")
        print("=" * 72)
        print("A run is suggested if it is DOMINATED BY ANOTHER ACTUAL RUN in the same (task,split) bucket,")
        print("i.e., another run is >= on top1/top5/top10/top50 and strictly better on at least one.\n")

        any_suggested = False
        to_delete = []

        for task in TASKS:
            for split in SPLITS:
                bucket = (task, split)
                bucket_runs = runs[bucket]
                if not bucket_runs:
                    continue

                dominated = []
                for r in bucket_runs:
                    r_vec = r["metrics"]
                    is_dom = False
                    for o in bucket_runs:
                        if o is r:
                            continue
                        if dominates(o["metrics"], r_vec):
                            is_dom = True
                            break
                    if is_dom:
                        dominated.append(r)

                if dominated:
                    any_suggested = True
                    print(f"\nBucket: {task.upper()} / {split.replace('_',' ').title()}")
                    print("  Candidates to delete (dominated by some other run):")
                    dominated.sort(
                        key=lambda x: (
                            x["metrics"]["top10"],
                            x["metrics"]["top50"],
                            x["metrics"]["top5"],
                            x["metrics"]["top1"],
                        )
                    )
                    for r in dominated:
                        name = os.path.basename(r["run_dir"])
                        m = r["metrics"]
                        print(
                            f'    - "{name}"  (top1={m["top1"]:.2f}%, top5={m["top5"]:.2f}%, '
                            f'top10={m["top10"]:.2f}%, top50={m["top50"]:.2f}%, avg={m["avg"]:.2f}%)'
                        )
                        to_delete.append(r["run_dir"])

        if not any_suggested:
            print("No dominated runs found (nothing is strictly worse than another run across all metrics).")

        # Deletion: requires BOTH toggles
        if DELETE_SUGGESTED and to_delete:
            if DELETE_CONFIRM_TOKEN != "YES_DELETE":
                print("\n[SAFETY] DELETE_SUGGESTED=True but DELETE_CONFIRM_TOKEN != 'YES_DELETE'.")
                print("[SAFETY] Not deleting anything. To delete, set DELETE_CONFIRM_TOKEN = 'YES_DELETE'.")
            else:
                for path in sorted(set(to_delete)):
                    shutil.rmtree(path)
                    print(f"[DELETED] {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
