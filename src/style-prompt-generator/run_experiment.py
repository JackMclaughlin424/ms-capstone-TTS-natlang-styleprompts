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
    compute_bertscore, compute_meteor, compute_chrf, compute_rouge, compute_tag_f1,
    compute_dist, compute_pred_semantic_sim, _flatten
)


from baseline import load_llm, build_system_prompt, build_user_prompt, batch_query_llm, LLM_REPO
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


def build_fewshot_set(train_ds, shuffled,cfg,num_few_shot):
        # build the full chain pool grouped by source to enable proportional sampling
    conv_id_to_chains: dict = {}
    for chain in train_ds._chains:
        cid = chain[-1].get("conv_id")
        conv_id_to_chains.setdefault(cid, []).append(chain)

    ordered_chains = []
    for conv_id in shuffled:
        if conv_id in conv_id_to_chains:
            ordered_chains.extend(conv_id_to_chains[conv_id])

    # group by source so we can mirror the training set's source proportions
    chains_by_source: dict = {}
    for chain in ordered_chains:
        src = str(chain[-1].get("source", "unknown")).lower()
        chains_by_source.setdefault(src, []).append(chain)

    total = len(ordered_chains)
    rng = np.random.default_rng(cfg["seed"])
    few_shot_chains = []
    allocated = 0
    sources = sorted(chains_by_source)  # sorted for deterministic rng consumption order
    for i, src in enumerate(sources):
        pool = chains_by_source[src]
        # give the last source any remaining slots to avoid rounding under-allocation
        if i == len(sources) - 1:
            n = max(0, num_few_shot - allocated)

        else:
            n = round(num_few_shot * len(pool) / total)
        n = min(n, len(pool))
        idxs = rng.choice(len(pool), size=n, replace=False)
        few_shot_chains.extend(pool[j] for j in idxs)
        allocated += n

    source_counts = {src: sum(1 for c in few_shot_chains if str(c[-1].get("source","")).lower() == src) for src in sources}
    log.info(f"Baseline: {len(few_shot_chains)} few-shot chains sampled with seed={cfg['seed']}  per-source={source_counts}  (num_turns={cfg['num_turns']})")

    return few_shot_chains


def run_baseline_for_trial(cfg, shuffled, test_chains_by_source, run, device, global_step):
    """
    Run the few-shot LLM baseline on the pre-built test set for one trial.
    Few-shot examples are the first num_few_shot chains drawn from the
    training pool in the trial's already-shuffled conv_id order, built at
    the same num_turns as the trial so the prompt context lengths match.
    """
    num_few_shot         = cfg.get("num_few_shot", 25)
    max_new_tokens       = cfg.get("max_new_tokens", 80)
    inference_batch_size = cfg.get("inference_batch_size", 8)
    llm_repo             = cfg.get("llm_repo", LLM_REPO)
    device_str           = str(device)

    train_ds = ConvoStyleDataset(
        h5_path=cfg["h5_path"],
        meta_path=cfg["meta_path"],
        meta_columns=["transcription", "text_description", "conv_id", "source"],
        num_turns=int(cfg["num_turns"]),
        max_len_sec=float(cfg["max_len_sec"]),
        allowed_conv_ids=set(shuffled),
    )

    few_shot_chains = build_fewshot_set(train_ds, shuffled,cfg,num_few_shot)
    
    log.info(f"Baseline: loading LLM ({llm_repo}) on {device_str}...")
    tokenizer, llm = load_llm(device_str, repo=llm_repo)

    # build the system prompt once since the few-shot pool is the same for all queries
    system_prompt = build_system_prompt(few_shot_chains)

    for src, chains in test_chains_by_source.items():
        full_prompts   = [f"{system_prompt}\n\n---\n\n{build_user_prompt(c)}" for c in chains]
        ground_truths  = [(c[-1].get("text_description") or "").strip() for c in chains]

        log.info(f"Baseline/{src}: running inference on {len(full_prompts)} chains...")
        t0_infer = time.time()
        predictions = batch_query_llm(
            tokenizer, llm, full_prompts, device_str,
            max_new_tokens=max_new_tokens,
            batch_size=inference_batch_size,
        )
        inference_time = time.time() - t0_infer
        log.info(f"Baseline/{src}: inference_time={inference_time:.1f}s")


        for i, (pred, ref, chain) in enumerate(zip(predictions[:3], ground_truths[:3], chains[:3])):
            txt = " | ".join(t.get("transcription", "") for t in chain if t.get("transcription"))
            log.info(f"  [Baseline/{src} Sample {i+1}]")
            log.info(f"    Dialogue : {txt}")
            log.info(f"    Predicted: {pred}")
            log.info(f"    Reference: {ref}")

        bs_metrics    = compute_bertscore(predictions, ground_truths, device=device_str)
        met_metrics   = compute_meteor(predictions, ground_truths)
        chrf_metrics  = compute_chrf(predictions, ground_truths)
        rouge_metrics = compute_rouge(predictions, ground_truths)
        tag_metrics   = compute_tag_f1(predictions, ground_truths, src)
        div_metrics  = compute_dist(predictions)
        psem_metrics = compute_pred_semantic_sim(predictions, device=device_str)


        all_metrics = {**_flatten(bs_metrics), **_flatten(met_metrics), **_flatten(chrf_metrics), **_flatten(rouge_metrics), **_flatten(tag_metrics), **div_metrics, **psem_metrics}


        summary = {f"baseline/{src}/{k}": v for k, v in all_metrics.items()}
        summary[f"baseline/{src}/inference_time_s"] = inference_time


        log.info(
            f"Baseline/{src}  bertscore_f1={all_metrics['bertscore_f1']:.4f}  "
            f"meteor={all_metrics['meteor']:.4f}  chrf={all_metrics['chrf']:.4f}  "
            f"tag_f1={all_metrics['tag_f1_overall']:.4f}"
        )
        run.summary.update(summary)
        wandb_log(summary, step=global_step, run=run)

    del llm, tokenizer
    gc.collect()
    if device_str == "cuda":
        torch.cuda.empty_cache()



def run_experiment_trial(cfg, trainval_ids, test_chains_by_source, run, device):

    test_metrics, global_step, training_time, inference_time = _train_final_and_eval_test(
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
        wandb_log(test_summary, step=global_step, run=run)

    time_summary = {"trial/training_time_s": training_time}
    log.info(f"Trial  training_time={training_time:.1f}s  Tot Inference time={inference_time:.1f}s")
    run.summary.update(time_summary)
    wandb_log(time_summary, step=global_step, run=run)

    return global_step





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

            # expand shared param into per-encoder keys
            if param_name == "num_unfrozen_embedder_layers":
                cfg["num_unfrozen_bert"]  = param_value
                cfg["num_unfrozen_wavlm"] = param_value
            else:   
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
                sample_rate=int(cfg["sample_rate"]),
                max_len_sec=float(cfg["max_len_sec"]),
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

            global_step = run_experiment_trial(cfg, trainval_ids, test_chains_by_source, run, device)

            run_baseline_for_trial(cfg, shuffled, test_chains_by_source, run, device, global_step)

            run.finish()

            gc.collect()
            torch.cuda.empty_cache()





if __name__ == "__main__":
    main()
