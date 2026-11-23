import json
import os
import time
from typing import Any, Dict, Optional

import torch


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class ExperimentLogger:
    """Writes one training run to its own folder:

        <root>/<task_name>/<timestamp>__split=<split_mode>__seed=<seed>/
            config.json          all training arguments
            split.json           the sentence split used
            metrics.jsonl        one line per epoch
            best_checkpoint.pt   checkpoint with the best validation metric
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
        higher_is_better: bool = True,
    ):
        run_id = "__".join([
            time.strftime("%Y-%m-%d_%H-%M-%S"),
            f"split={split_info.get('split_mode', 'unknown')}",
            f"seed={config.get('seed', 'na')}",
        ])
        self.run_dir = os.path.join(root, task_name, run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None

        write_json(os.path.join(self.run_dir, "config.json"), config)
        write_json(os.path.join(self.run_dir, "split.json"), split_info)
        self.metrics_path = os.path.join(self.run_dir, "metrics.jsonl")

    @property
    def best_checkpoint_path(self) -> str:
        return os.path.join(self.run_dir, "best_checkpoint.pt")

    def log_epoch(self, epoch: int, train_stats: Dict[str, Any], val_stats: Dict[str, Any]) -> None:
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "train": train_stats, "val": val_stats}) + "\n")

    def save_last(self, checkpoint: Dict[str, Any]) -> None:
        torch.save(checkpoint, os.path.join(self.run_dir, "last_checkpoint.pt"))

    def maybe_save_best(self, epoch: int, metric_value: float, checkpoint: Dict[str, Any]) -> bool:
        """Save ``checkpoint`` as the best one if ``metric_value`` improves on the best so far."""
        if self.best_value is not None:
            improved = metric_value > self.best_value if self.higher_is_better else metric_value < self.best_value
            if not improved:
                return False

        self.best_value = float(metric_value)
        self.best_epoch = int(epoch)
        torch.save(checkpoint, self.best_checkpoint_path)
        write_json(os.path.join(self.run_dir, "best_summary.json"), {
            "metric": self.metric_name,
            "higher_is_better": self.higher_is_better,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
        })
        return True
