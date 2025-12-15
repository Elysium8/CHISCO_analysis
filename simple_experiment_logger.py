# simple_experiment_logger.py
import os
import json
import time
from typing import Any, Dict, Optional

import torch


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _write_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class SimpleExperimentLogger:
    """
    Minimal paper-friendly logger.

    Creates:
      experiments/<task_name>/<run_id>/
        config.json
        split.json
        metrics.jsonl
        best_checkpoint.pt
        best_summary.json
        last_checkpoint.pt
    """
    def __init__(
        self,
        root: str,
        task_name: str,
        config: Dict[str, Any],
        split_info: Dict[str, Any],
        metric_name: str,
        higher_is_better: bool,
        run_tag: Optional[str] = None,
    ):
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        split_mode = split_info.get("split_mode", "unknown")
        seed = config.get("RANDOM_SEED", "na")

        run_id_parts = [ts, f"split={split_mode}", f"seed={seed}"]
        if run_tag:
            run_id_parts.append(str(run_tag))
        run_id = "__".join(run_id_parts)

        self.run_dir = os.path.join(root, task_name, run_id)
        _ensure_dir(self.run_dir)

        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None

        _write_json(os.path.join(self.run_dir, "config.json"), config)
        _write_json(os.path.join(self.run_dir, "split.json"), split_info)

        self.metrics_path = os.path.join(self.run_dir, "metrics.jsonl")

    def log_epoch(self, epoch: int, train_stats: Dict[str, Any], val_stats: Dict[str, Any]) -> None:
        rec = {"epoch": epoch, "train": train_stats, "val": val_stats}
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def save_last(self, checkpoint: Dict[str, Any]) -> None:
        torch.save(checkpoint, os.path.join(self.run_dir, "last_checkpoint.pt"))

    def _is_better(self, v: float) -> bool:
        if self.best_value is None:
            return True
        return (v > self.best_value) if self.higher_is_better else (v < self.best_value)

    def maybe_save_best(self, epoch: int, metric_value: float, checkpoint: Dict[str, Any]) -> bool:
        if not self._is_better(metric_value):
            return False

        self.best_value = float(metric_value)
        self.best_epoch = int(epoch)

        torch.save(checkpoint, os.path.join(self.run_dir, "best_checkpoint.pt"))
        _write_json(
            os.path.join(self.run_dir, "best_summary.json"),
            {
                "metric": self.metric_name,
                "higher_is_better": self.higher_is_better,
                "best_value": self.best_value,
                "best_epoch": self.best_epoch,
            },
        )
        return True
