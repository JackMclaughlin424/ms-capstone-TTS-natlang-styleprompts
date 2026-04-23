"""
sweep.py  --config path/to/sweep_base_config.json
          --sweep_values path/to/sweep_values.json
          [options]

Runs a W&B hyperparameter sweep where each trial performs N-fold cross-validation.
Conversations are split at the conv_id level (same anti-leakage logic as train.py).

Usage
-----
# Start a new sweep and run one agent:
    python sweep.py --config default_sweep_config.json --sweep_values default_sweep_values.json --n_folds 5 --count 20

# Join an existing sweep from a second machine:
    python sweep.py --config default_sweep_config.json --sweep_values default_sweep_values.json --sweep_id <SHORT_ID> --count 10
"""


import argparse
import logging
from copy import deepcopy
import gc
import numpy as np
import pandas as pd
import torch
import wandb
import json
import os
import time
import itertools

from model.train_helpers import (
    load_config, apply_overrides, wandb_log, assert_no_test_leakage, 
)

from sweep import _train_final_and_eval_test
from dataset.ConvoStyleDataset import ConvoStyleDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# write our logs to a file independent of wandb's stderr capture
_log_file = os.environ.get("SLURM_JOB_LOG", "sweep_run.log")
_fh = logging.FileHandler(_log_file, mode="a")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(_fh)


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)





# Entry point

def main():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments over a single swept parameter."
    )
    parser.add_argument("--config",            required=True,
                        help="Base config JSON with fixed hyperparameters.")
    parser.add_argument("--experiment_config", required=True,
                        help="JSON with one key mapping to a list of values to sweep.")
    parser.add_argument("--num_trials",        type=int, default=3,
                        help="Number of independent training runs per parameter value.")
    parser.add_argument("--override",          nargs="*", metavar="KEY=VALUE",
                        help="Override base config fields (same syntax as train.py).")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    apply_overrides(base_cfg, args.override)

    with open(args.experiment_config) as f:
        experiment_config = json.load(f)

    if len(experiment_config) != 1:
        raise ValueError(f"experiment_config must contain exactly one parameter, got: {list(experiment_config.keys())}")

    param_name, param_values = next(iter(experiment_config.items()))
    log.info(f"Experiment parameter: {param_name}  values: {param_values}  trials_per_value: {args.num_trials}")

    # master rng derives per-trial seeds so each trial is reproducible but distinct
    master_rng = np.random.default_rng(base_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for param_value in param_values:
        for trial_idx in range(args.num_trials):
            trial_seed = int(master_rng.integers(0, 2**31))
            cfg = deepcopy(base_cfg)
            cfg[param_name] = param_value
            cfg["seed"]     = trial_seed

            log.info(f"=== {param_name}={param_value}  trial={trial_idx+1}/{args.num_trials}  seed={trial_seed} ===")
            log.info(f"Run config: {json.dumps(cfg, indent=2, default=str)}")

            run = wandb.init(
                project=cfg["wandb_project"],
                entity=cfg.get("wandb_entity"),
                config={
                    **cfg,
                    "experiment_param": param_name,
                    "experiment_value": param_value,
                    "trial_idx":        trial_idx,
                    "num_trials":       args.num_trials,
                },
                settings=wandb.Settings(console="off", init_timeout=300),
            )

            test_chains_by_source, test_conv_ids = ConvoStyleDataset.make_fixed_test_split(
                h5_path=cfg["h5_path"],
                meta_path=cfg["meta_path"],
                meta_columns=["transcription", "text_description", "source"],
                sample_rate=cfg["sample_rate"],
                max_len_sec=cfg["max_len_sec"],
                num_turns=int(cfg["num_turns"]),
            )


            meta         = pd.read_parquet(cfg["meta_path"], columns=["conv_id"])
            trainval_arr = np.array([c for c in meta["conv_id"].unique() if c not in test_conv_ids])
            assert_no_test_leakage(set(trainval_arr), test_conv_ids)

            n_test = sum(len(v) for v in test_chains_by_source.values())
            log.info(
                f"Split sizes  trainval_conv_ids={len(trainval_arr)}  test_chains={n_test}  "
                + "  ".join(f"{src}={len(c)}" for src, c in test_chains_by_source.items())
            )

            # shuffle trainval with the trial seed so each trial sees a different ordering
            rng      = np.random.default_rng(trial_seed)
            shuffled = trainval_arr.copy()
            rng.shuffle(shuffled)
            trainval_ids = set(shuffled)

            test_metrics, _ = _train_final_and_eval_test(
                cfg, trainval_ids, test_chains_by_source, run, device
            )

            for src, src_m in test_metrics.items():
                log.info(
                    f"Test/{src}  bertscore_f1={src_m['bertscore_f1']:.4f}  "
                    f"meteor={src_m['meteor']:.4f}  chrf={src_m['chrf']:.4f}  "
                    f"tag_f1={src_m['tag_f1_overall']:.4f}"
                )
                test_summary = {f"test/{src}/{k}": v for k, v in src_m.items()}
                run.summary.update(test_summary)
                wandb_log(test_summary, run=run)

            run.finish()
            gc.collect()
            torch.cuda.empty_cache()





if __name__ == "__main__":
    main()
