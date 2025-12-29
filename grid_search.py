"""
True grid search launcher for EEG→text experiments (single context).

This script:
- uses ONE task and ONE split mode
- runs a TRUE grid search over reduced hyperparameters
- varies temperature, weight decay, and learning rate
- runs updated_train_eeg_to_text_embed.py for each combination

Usage:
  python run_true_grid_search.py
"""

import subprocess
import itertools
from copy import deepcopy
import sys

PYTHON = sys.executable
TRAIN_SCRIPT = "updated_train_eeg_to_text_embed.py"

# -------------------------------
# Fixed base configuration
# -------------------------------
BASE_ARGS = {
    "--task": "read",                # FIXED context
    "--split-mode": "iid",  # FIXED context
    "--batch-size": 64,
    "--num-epochs": 30,
    "--learning-rate": 5e-4,  # will be overridden
    "--weight-decay": 1e-4,   # will be overridden
    "--temperature": 0.07,    # will be overridden
    "--latent-dim": 256,
    "--hidden-channels": 256,
    "--final-hidden": 256,
    "--n-blocks": 5,
    "--pooling": "mean",
    "--use-eegnet": True,
    "--use-spatial-attn": True,
    "--use-subject-layer": True,
    "--use-residual-blocks": True,
    "--seed": 42,
}

# -------------------------------
# True grid (reduced + sensible)
# -------------------------------
GRID = {
    "--task": ["read", "imagine"],
    "--temperature": [0.05, 0.07, 0.10],
    "--weight-decay": [1e-5, 1e-4, 5e-4],
    "--learning-rate": [3e-4, 5e-4, 7e-4],
}

# -------------------------------
# Helper
# -------------------------------
def run_experiment(arg_dict):
    cmd = [PYTHON, TRAIN_SCRIPT]

    for k, v in arg_dict.items():
        if isinstance(v, bool):
            if v:
                cmd.append(k)
            else:
                cmd.append(k.replace("--use-", "--no-use-"))
        else:
            cmd.extend([k, str(v)])

    print("\nLaunching:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

# -------------------------------
# Main loop (true Cartesian grid)
# -------------------------------
def main():
    keys = list(GRID.keys())
    values = list(GRID.values())

    print("\n" + "=" * 80)
    print("TRUE GRID SEARCH")
    print(f"Task: {BASE_ARGS['--task']} | Split mode: {BASE_ARGS['--split-mode']}")
    print("=" * 80)

    for combo in itertools.product(*values):
        args = deepcopy(BASE_ARGS)
        for k, v in zip(keys, combo):
            args[k] = v

        run_experiment(args)

if __name__ == "__main__":
    main()
