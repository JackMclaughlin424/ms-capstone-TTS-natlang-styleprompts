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
    load_config, apply_overrides, set_seed,
    build_model, build_optimizer_and_scheduler,
    wandb_log, compute_bertscore, compute_meteor, compute_chrf,
)
from train import run_epoch
from dataset.ConvoStyleDataset import ConvoStyleDataset, collate_pad
from torch.utils.data import DataLoader

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




# Data helpers 

def _get_conv_ids(meta_path: str, data_source: str = "both") -> np.ndarray:
    meta = pd.read_parquet(meta_path)
    if data_source != "both":
        meta = meta[meta["source"] == data_source]
    return np.array(meta["conv_id"].unique())


def _carve_data_splits(all_conv_ids: np.ndarray, base_cfg: dict, data_source: str = "both"):
    if data_source != "both":
        other_source  = "styletalk" if data_source == "expresso" else "expresso"
        other_ids     = _get_conv_ids(base_cfg["meta_path"], other_source)
        max_test      = base_cfg.get("max_test_convos")
        if max_test is not None:
            rng_test  = np.random.default_rng(base_cfg["seed"])
            rng_test.shuffle(other_ids)
            other_ids = other_ids[:max_test]
        test_ids     = set(other_ids)
        trainval_ids = all_conv_ids
        log.info(
            f"{len(trainval_ids)} {data_source} conversations for train/val  |  "
            f"{len(test_ids)} {other_source} conversations for test (cross-dataset)"
        )
    else:
        # carve out a 10% held-out test split before any fold construction
        rng_split    = np.random.default_rng(base_cfg["seed"])
        shuffled_all = all_conv_ids.copy()
        rng_split.shuffle(shuffled_all)
        n_test       = max(1, int(len(shuffled_all) * 0.10))
        test_ids     = set(shuffled_all[:n_test])
        trainval_ids = shuffled_all[n_test:]
        log.info(
            f"{len(all_conv_ids)} unique conversations → "
            f"{len(trainval_ids)} train/val  |  {len(test_ids)} test (held-out)"
        )

    return trainval_ids, test_ids



def _build_fold_loaders(cfg: dict, train_ids: set, val_ids: set):
    ds_kwargs = dict(
        h5_path=cfg["h5_path"],
        meta_path=cfg["meta_path"],
        meta_columns=["transcription", "text_description"],
        sample_rate=cfg["sample_rate"],
        num_turns=cfg["num_turns"],
        max_len_sec=cfg["max_len_sec"],
    )
    train_ds = ConvoStyleDataset(**ds_kwargs, allowed_conv_ids=train_ids)
    val_ds   = ConvoStyleDataset(**ds_kwargs, allowed_conv_ids=val_ids)

    loader_kw = dict(collate_fn=collate_pad, num_workers=cfg["num_workers"], pin_memory=True)
    g = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=g, **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, **loader_kw)
    log.info(f"  {len(train_ds)} train chains  |  {len(val_ds)} val chains")
    return train_loader, val_loader


# Single-fold training 

def _train_fold(
    cfg: dict,
    train_ids: set,
    val_ids: set,
    fold_idx: int,
    sweep_run,              # the W&B run owned by the sweep agent
    device: torch.device,
    global_step: int
) -> dict:
    """Train one fold; return dict with best-epoch metrics."""
    set_seed(cfg["seed"] + fold_idx)  # each fold gets a distinct but reproducible seed

    train_loader, val_loader = _build_fold_loaders(cfg, train_ids, val_ids)
    model = build_model(cfg, device, log)

    total_steps = len(train_loader) * cfg["num_epochs"]
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, total_steps, log)
    

    best: dict = {"val_loss": float("inf"), "epoch": -1}
    
    fp = f"fold_{fold_idx}"

    patience         = cfg.get("early_stopping_patience", 3)
    min_delta        = cfg.get("early_stopping_min_delta", 1e-4)
    MIN_EPOCH        = 5
    epochs_no_improve = 0

    fold_start_time = time.time()
    for epoch in range(cfg["num_epochs"]):
        train_loss, global_step = run_epoch(
            model, train_loader, optimizer, scheduler,
            device, cfg, epoch, global_step, wandb_run=None, log_handler=log
            , is_train=True, use_tqdm=False
        )

        if (epoch + 1) % cfg["eval_every_n_epochs"] != 0:
            continue

        val_loss, _ = run_epoch(
            model, val_loader, optimizer, scheduler,
            device, cfg, epoch, global_step, wandb_run=None, log_handler=log
            , is_train=False, use_tqdm=False
        )

        wandb_log({
            f"{fp}/train_loss": train_loss,
            f"{fp}/val_loss":   val_loss,
            "epoch":            epoch,
        }, step=global_step, run=sweep_run)

        if val_loss < best["val_loss"] - min_delta:
            best["val_loss"] = val_loss
            best["epoch"]    = epoch
            epochs_no_improve = 0
        elif patience > 0 and epoch >= MIN_EPOCH:
            epochs_no_improve += 1

        if patience > 0 and epoch >= MIN_EPOCH and epochs_no_improve >= patience:
            log.info(f"  Fold {fold_idx}: early stop at epoch {epoch} (no improvement for {patience} evals)")
            break

        gc.collect()

    elapsed = time.time() - fold_start_time
    fmt = lambda s: f"{int(s)//60:02d}:{int(s)%60:02d}"
    log.info(
        f"Fold {fold_idx}: Trained for {epoch} epochs. Total time: {fmt(elapsed)}"
        f"\nBeginning validation..."
    )

    # compute text-quality metrics once, on the final model state
    model.eval()
    all_preds, all_refs, all_texts, all_vecs = [], [], [], []

    with torch.no_grad():
        for batch in val_loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                audio       = batch["audio"].to(device)
                lengths     = batch["lengths"].to(device)
                text_only   = batch["text_only"].to(device)
                texts       = batch["transcription"]
                speaker_ids = batch["speaker_id"]
                targets     = batch["text_description"]
                if cfg["num_turns"] == 0:
                    audio     = torch.zeros_like(audio)
                    text_only = torch.ones_like(text_only)
                ctx = model.scfa(audio, lengths, texts, speaker_ids, text_only)
                vec = model.pooler(ctx)
                del ctx
                all_vecs.append(vec.float().detach().cpu())  # collect before delete
                preds = model.style_generator.generate(vec)
                del vec
            
            all_preds.extend(preds)
            all_refs.extend([chain[-1] for chain in targets])
            all_texts.extend(texts)

    # free GPU memory before the next fold loads a fresh model
    del train_loader, val_loader
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    # collapse diagnostics on pooled dialogue vectors
    vecs = torch.cat(all_vecs, dim=0)               # (N, 4*d_model)
    vec_std      = vecs.std(dim=0).mean().item()     # near 0 = collapsed
    vec_norm_cv  = (vecs.norm(dim=-1).std() /
                    vecs.norm(dim=-1).mean()).item()  # coeff of variation of norms

    # pairwise cosine similarity — mean off-diagonal → 1.0 = fully collapsed
    normed   = torch.nn.functional.normalize(vecs, dim=-1)
    sim_mat  = normed @ normed.T
    n        = sim_mat.shape[0]
    off_diag = sim_mat[~torch.eye(n, dtype=torch.bool)].mean().item()

    wandb_log({
        f"{fp}/vec_std":          vec_std,
        f"{fp}/vec_norm_cv":      vec_norm_cv,
        f"{fp}/mean_cosine_sim":  off_diag,   # target: low (< 0.5); alarm: > 0.9
    }, step=global_step, run=sweep_run)
    del all_vecs, vecs


    bs  = compute_bertscore(all_preds, all_refs, device=str(device))
    met = compute_meteor(all_preds, all_refs)
    chrf = compute_chrf(all_preds, all_refs)

    wandb_log({
        f"{fp}/bertscore_f1": bs["bertscore_f1_mean"],
        f"{fp}/meteor":       met["meteor_mean"],
        f"{fp}/chrf":         chrf["chrf_mean"],
    }, step=global_step, run=sweep_run)

    for i, (pred, ref, txt) in enumerate(zip(all_preds[:3], all_refs[:3], all_texts[:3])):
        log.info(f"  [Fold {fold_idx} Sample {i+1}]")
        log.info(f"    Dialogue : {txt}")
        log.info(f"    Predicted: {pred}")
        log.info(f"    Reference: {ref}")

    gc.collect()
    torch.cuda.empty_cache()  # reclaim BERTScore model memory before next fold loads
    
    return {
        "val_loss":     best["val_loss"],
        "best_epoch":   best["epoch"],
        "bertscore_f1": bs["bertscore_f1_mean"],
        "meteor":       met["meteor_mean"],
        "chrf":         chrf["chrf_mean"],
    }, global_step



def _train_final_and_eval_test(
    cfg: dict,
    trainval_ids: set,
    test_ids: set,
    sweep_run,
    device: torch.device,
    global_step: int = 0,
) -> dict:
    """Train on all train/val data, evaluate generation metrics on held-out test split."""
    set_seed(cfg["seed"])
    train_loader, test_loader = _build_fold_loaders(cfg, trainval_ids, test_ids)
    model = build_model(cfg, device, log)

    total_steps = len(train_loader) * cfg["num_epochs"]
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, total_steps, log)
   

    for epoch in range(cfg["num_epochs"]):
        _, global_step = run_epoch(
            model, train_loader, optimizer, scheduler, 
            device, cfg, epoch, global_step, wandb_run=None, log_handler=log
            , is_train=True, use_tqdm=False
        )

    model.eval()
    test_loss, _ = run_epoch(
        model, test_loader, optimizer, scheduler, 
        device, cfg, 0, global_step, wandb_run=None, log_handler=log
        , is_train=False, use_tqdm=False
    )

    all_preds, all_refs, all_texts = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                audio       = batch["audio"].to(device)
                lengths     = batch["lengths"].to(device)
                text_only   = batch["text_only"].to(device)
                texts       = batch["transcription"]
                speaker_ids = batch["speaker_id"]
                targets     = batch["text_description"]
                if cfg["num_turns"] == 0:
                    audio     = torch.zeros_like(audio)
                    text_only = torch.ones_like(text_only)
                
                ctx   = model.scfa(audio, lengths, texts, speaker_ids, text_only)
                vec   = model.pooler(ctx)
                del ctx
                preds = model.style_generator.generate(vec)
                del vec


                all_preds.extend(preds)
                all_refs.extend([chain[-1] for chain in targets])
                all_texts.extend(texts)

    # memory management
    del train_loader, test_loader
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    bs  = compute_bertscore(all_preds, all_refs, device=str(device))
    met = compute_meteor(all_preds, all_refs)
    chrf = compute_chrf(all_preds, all_refs)

    wandb_log({
        f"test/bertscore_f1": bs["bertscore_f1_mean"],
        f"test/meteor":       met["meteor_mean"],
        f"test/chrf":         chrf["chrf_mean"],
    }, step=global_step, run=sweep_run)


    for i, (pred, ref, txt) in enumerate(zip(all_preds[:3], all_refs[:3], all_texts[:3])):
        log.info(f"  [Test Sample {i+1}]")
        log.info(f"    Dialogue : {txt}")
        log.info(f"    Predicted: {pred}")
        log.info(f"    Reference: {ref}")

    gc.collect()
    torch.cuda.empty_cache()  # reclaim BERTScore model memory before next fold loads
    
    return {
        "test_loss":     test_loss,
        "bertscore_f1":  bs["bertscore_f1_mean"],
        "meteor":        met["meteor_mean"],
        "chrf":          chrf["chrf_mean"],
    }, global_step



# Sweep function 

def _make_sweep_fn(base_cfg: dict, n_folds: int):

    """Return the callable that W&B's agent invokes for each hyperparameter trial."""

    def sweep_fn():
        gc.collect()
        torch.cuda.empty_cache()
        run = wandb.init(settings=wandb.Settings(console="off"))
        cfg = deepcopy(base_cfg)

        for key, val in run.config.items():
            if key in cfg:
                cfg[key] = val

        # expand shared sweep param into per-encoder keys
        if "num_unfrozen_embedder_layers" in run.config:
            n = run.config["num_unfrozen_embedder_layers"]
            cfg["num_unfrozen_bert"]  = n
            cfg["num_unfrozen_wavlm"] = n

        run.config.update({"n_folds": n_folds}, allow_val_change=True)

        data_source      = cfg.get("data_source", "both")
        raw_conv_ids     = _get_conv_ids(cfg["meta_path"], data_source)
        trainval_arr, test_ids = _carve_data_splits(raw_conv_ids, cfg, data_source)

        rng      = np.random.default_rng(cfg["seed"])
        shuffled = trainval_arr.copy()
        rng.shuffle(shuffled)
        folds = np.array_split(shuffled, n_folds)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        global_step = 0
        fold_metrics = []
        for fold_idx in range(n_folds):
            log.info(f"=== Fold {fold_idx + 1}/{n_folds} ===")
            val_ids   = set(folds[fold_idx])
            train_ids = set(np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx]))
            metrics, global_step   = _train_fold(cfg, train_ids, val_ids, fold_idx, run, device, global_step)
            fold_metrics.append(metrics)
            log.info(
                f"Fold {fold_idx + 1}  val_loss={metrics['val_loss']:.4f}  "
                f"bertscore_f1={metrics['bertscore_f1']:.4f}  meteor={metrics['meteor']:.4f}"
            )

        # aggregate across folds (the sweep optimises this)
        val_losses    = [m["val_loss"]     for m in fold_metrics]
        bert_f1s      = [m["bertscore_f1"] for m in fold_metrics]
        meteor_scores = [m["meteor"]       for m in fold_metrics]
        chrf_scores   = [m["chrf"]         for m in fold_metrics]

        # from training, average best epoch to stop at for test
        mean_best_epoch = int(round(np.mean([m["best_epoch"] for m in fold_metrics])))
        log.info(f"Mean best epoch across folds: {mean_best_epoch}")

        summary = {
            "cv/mean_val_loss":     float(np.mean(val_losses)),
            "cv/std_val_loss":      float(np.std(val_losses)),
            "cv/mean_bertscore_f1": float(np.mean(bert_f1s)),
            "cv/std_bertscore_f1":  float(np.std(bert_f1s)),
            "cv/mean_meteor":       float(np.mean(meteor_scores)),
            "cv/std_meteor":        float(np.std(meteor_scores)),
            "cv/mean_chrf":         float(np.mean(chrf_scores)),
            "cv/std_chrf":          float(np.std(chrf_scores)),
        }

        # held-out test evaluation (train on all trainval, eval on test) 
        log.info("=== Final model: training on all train/val folds for test eval ===")
        trainval_ids = set(np.concatenate(folds))
        test_metrics = _train_final_and_eval_test(cfg, trainval_ids, test_ids, run, device)
        log.info(
            f"Test  loss={test_metrics['test_loss']:.4f}  "
            f"bertscore_f1={test_metrics['bertscore_f1']:.4f}  "
            f"meteor={test_metrics['meteor']:.4f}"
            f"chrf={test_metrics["chrf"]:.4f}"
        )
        summary.update({
            "test/loss":         test_metrics["test_loss"],
            "test/bertscore_f1": test_metrics["bertscore_f1"],
            "test/meteor":       test_metrics["meteor"],
            "test/chrf":         test_metrics["chrf"],
        })

        run.summary.update(summary)
        wandb_log(summary, step=global_step, run=run)
        log.info(
            f"CV summary  mean_val_loss={summary['cv/mean_val_loss']:.4f} "
            f"(±{summary['cv/std_val_loss']:.4f})  "
            f"mean_bertscore_f1={summary['cv/mean_bertscore_f1']:.4f}"
        )
        run.finish()


    return sweep_fn


def _make_test_sweep_fn(base_cfg: dict, num_trials: int, trial_seeds: list):
    """Return the callable W&B invokes for each grid-search combo in test-experiment mode.
    Runs num_trials repetitions of _train_final_and_eval_test with independent split seeds."""

    def sweep_fn():
        gc.collect()
        torch.cuda.empty_cache()
        run = wandb.init(settings=wandb.Settings(console="off"))
        cfg = deepcopy(base_cfg)

        for key, val in run.config.items():
            if key in cfg:
                cfg[key] = val
        if "num_unfrozen_embedder_layers" in run.config:
            n = run.config["num_unfrozen_embedder_layers"]
            cfg["num_unfrozen_bert"]  = n
            cfg["num_unfrozen_wavlm"] = n

        run.config.update({"num_trials": num_trials}, allow_val_change=True)

        device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_source  = cfg.get("data_source", "both")
        all_conv_ids = _get_conv_ids(cfg["meta_path"], data_source)

        global_step   = 0
        trial_results = []
        for trial_idx, trial_seed in enumerate(trial_seeds):
            log.info(f"=== Trial {trial_idx + 1}/{num_trials} (seed={trial_seed}) ===")
            trial_cfg         = deepcopy(cfg)
            trial_cfg["seed"] = int(trial_seed)

            trainval_ids, test_ids = _carve_data_splits(all_conv_ids, trial_cfg, data_source)
            metrics, global_step = _train_final_and_eval_test(trial_cfg, trainval_ids, test_ids, run, device, global_step)
            
            trial_results.append(metrics)
            log.info(
                f"Trial {trial_idx + 1}: test_loss={metrics['test_loss']:.4f}  "
                f"bertscore_f1={metrics['bertscore_f1']:.4f}  meteor={metrics['meteor']:.4f}"
            )

        bert_f1s      = [m["bertscore_f1"] for m in trial_results]
        meteor_scores = [m["meteor"]        for m in trial_results]
        test_losses   = [m["test_loss"]     for m in trial_results]
        chrf_scores   = [m["chrf"]          for m in trial_results]

        summary = {
            "trials/mean_bertscore_f1": float(np.mean(bert_f1s)),
            "trials/std_bertscore_f1":  float(np.std(bert_f1s)),
            "trials/mean_meteor":       float(np.mean(meteor_scores)),
            "trials/std_meteor":        float(np.std(meteor_scores)),
            "trials/mean_test_loss":    float(np.mean(test_losses)),
            "trials/std_test_loss":     float(np.std(test_losses)),
            "trials/mean_chrf":         float(np.mean(chrf_scores)),
            "trials/std_chrf":          float(np.std(chrf_scores)),
        }
        run.summary.update(summary)
        wandb_log(summary, step=global_step, run=run)

        log.info(
            f"Trial summary: bertscore_f1={summary['trials/mean_bertscore_f1']:.4f} "
            f"(±{summary['trials/std_bertscore_f1']:.4f})  "
            f"meteor={summary['trials/mean_meteor']:.4f} "
            f"(±{summary['trials/std_meteor']:.4f})"
            f"chrf={summary["trials/mean_chrf"]:.4f}"
            f"(±{summary['trials/std_chrf']:.4f})"
        )
        run.finish()

    return sweep_fn



# Entry point

def main():
    parser = argparse.ArgumentParser(
        description="W&B hyperparameter sweep with N-fold cross-validation."
    )
    parser.add_argument("--config",       required=True,
                        help="Base config JSON with fixed hyperparameters (e.g. default_sweep_config.json).")
    parser.add_argument("--sweep_values", required=True,
                        help="W&B sweep search-space JSON (e.g. default_sweep_values.json).")
    parser.add_argument("--experiment_type", choices=["hyperparameter", "test"],
                        default="hyperparameter",
                        help="'hyperparameter' runs a Bayesian/random W&B sweep; "
                             "'test' runs a grid sweep where each combo is repeated num_trials times "
                             "with independent random train/test splits.")
    parser.add_argument("--num_trials",  type=int, default=5,
                        help="Number of independent split-seed trials per combo (test mode only).")
    parser.add_argument("--n_folds",     type=int, default=5,
                        help="Number of CV folds (hyperparameter mode only, default: 5).")
    parser.add_argument("--sweep_id",    default=None,
                        help="Short sweep ID to join an existing sweep instead of creating one.")
    parser.add_argument("--count",       type=int, default=None,
                        help="Max trials this agent will run (default: unlimited).")
    parser.add_argument("--override",    nargs="*", metavar="KEY=VALUE",
                        help="Override base config fields (same syntax as train.py).")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    base_cfg = apply_overrides(base_cfg, args.override)

    with open(args.sweep_values) as f:
        sweep_config = json.load(f)

    
    if args.experiment_type == "test":
        master_rng  = np.random.default_rng(base_cfg["seed"])
        trial_seeds = master_rng.integers(0, 2**31, size=args.num_trials).tolist()
        sweep_fn    = _make_test_sweep_fn(base_cfg, args.num_trials, trial_seeds)
    else:
        
        sweep_fn = _make_sweep_fn(base_cfg, args.n_folds)

    project = base_cfg["wandb_project"]
    entity  = base_cfg.get("wandb_entity")

    if args.sweep_id:
        sweep_id = args.sweep_id
        log.info(f"Joining existing sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(deepcopy(sweep_config), project=project, entity=entity)
        log.info(f"Created sweep: {sweep_id}")

    wandb.agent(sweep_id, function=sweep_fn, count=args.count,
                project=project, entity=entity)





if __name__ == "__main__":
    main()
