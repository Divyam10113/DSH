"""
Lightweight Shared Experiment Tracking.
Owned by Pranav (M4 - Integration).

Every run writes:
- <run_dir>/metrics.csv       one row per epoch (loss, macro-AUC, 12 per-label AUCs, lr, time)
- <run_dir>/config.json       full training config for reproducibility
- <leaderboard_csv>           one summary row per finished run/fold (shared team leaderboard)

Weights & Biases is used additionally when `use_wandb=True` and the package + API key are available
(Kaggle dev notebooks have internet; the offline submission notebook never imports this module).
"""

import json
import os
import time
from typing import Dict, Optional

import pandas as pd


class ExperimentTracker:
    def __init__(self, run_dir: str, config: Dict, leaderboard_csv: Optional[str] = None,
                 use_wandb: bool = False, run_name: Optional[str] = None):
        self.run_dir = run_dir
        self.config = config
        self.leaderboard_csv = leaderboard_csv
        self.run_name = run_name or os.path.basename(os.path.normpath(run_dir))
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)

        self.metrics_path = os.path.join(run_dir, "metrics.csv")
        self._wandb = None
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb.init(project="rsna-knee", name=self.run_name, config=config,
                                         resume="allow", id=self.run_name.replace("/", "-"))
            except Exception as e:  # never let tracking kill a training session
                print(f"[Tracker] W&B disabled: {e}")

    def log_epoch(self, metrics: Dict) -> None:
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **metrics}
        header = not os.path.exists(self.metrics_path)
        pd.DataFrame([row]).to_csv(self.metrics_path, mode="a", header=header, index=False)
        if self._wandb is not None:
            self._wandb.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

    def log_summary(self, summary: Dict) -> None:
        if self.leaderboard_csv:
            row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "run": self.run_name, **summary}
            os.makedirs(os.path.dirname(os.path.abspath(self.leaderboard_csv)), exist_ok=True)
            header = not os.path.exists(self.leaderboard_csv)
            pd.DataFrame([row]).to_csv(self.leaderboard_csv, mode="a", header=header, index=False)
        if self._wandb is not None:
            self._wandb.summary.update(summary)
            self._wandb.finish()
