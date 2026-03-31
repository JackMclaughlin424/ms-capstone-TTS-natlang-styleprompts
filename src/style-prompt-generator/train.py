"""
train.py  --config path/to/config.json

Trains the SCFAWithStyleHead pipeline end-to-end (or partially frozen),
reading all hyperparameters from a JSON config file.
"""

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    Wav2Vec2FeatureExtractor,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from ConvoStyleDataset import ConvoStyleDataset, collate_pad
from DialogueEncoder import (
    DualModalityEmbedder,
    SCFA,
    DialoguePooler,
    SelfAttentivePooling,
    ModalityEncoder,
)
from StylePromptGenerator import (
    StylePromptHead,
    StylePromptGenerator,
    SCFAWithStyleHead,
    TINYLLAMA_REPO,
    TINYLLAMA_DIM,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# defaults

DEFAULTS: Dict[str, Any] = {
    # data
    "h5_path":               None,        # required
    "meta_path":             None,        # required
    "output_dir":            "runs/exp",
    "num_turns":             5,           # 0 means text-only (script-only mode, no audio)
    "max_len_sec":           None,        # trim/pad audio to this duration

    # model dimensions
    "d_model":               768,         # must be divisible by 3 (192, 384, or 768)
    "num_ctx_layers":        2,           # ContextAwareTransformer depth
    "num_spk_layers":        2,           # SpeakerAwareTransformer depth
    "dim_feedforward":       2048,
    "nhead":                 3,           # must divide d_model AND d_model//3

    # prefix / LLM
    "num_prefix_tokens":     10,          # K in StyleCap notation
    "num_mapping_layers":    8,
    "mapping_nhead":         8,
    "system_prompt":         "",        # "" | "Speaking style:" | custom string
    "max_prompt_tokens": 128,   # system prompt text -- used at both train and inference
    "max_style_desc_tokens": 128,  # ground-truth target style descriptions -- training only
    "max_new_tokens": 80,       # generation budget -- inference only

    # freezing
    "num_unfrozen_bert":     0,           # how many BERT encoder layers to unfreeze (from top)
    "num_unfrozen_wavlm":    0,           # how many WavLM encoder layers to unfreeze (from top)

    # pooling
    "dialogue_pooler":       "last", # "attentive" | "last"

    # training
    "batch_size":            8,
    "num_epochs":            20,
    "learning_rate":         5e-4,
    "weight_decay":          1e-2,
    "grad_clip":             1.0,         # max gradient norm; None to disable
    "warmup_ratio":          0.1,         # fraction of total steps used for LR warmup
    "lr_schedule":           "cosine",    # "cosine" | "linear" | "constant"
    "dropout":               0.1,

    # validation / checkpointing
    "val_split":             0.1,         # fraction of data held out for validation
    "eval_every_n_epochs":   1,
    "save_every_n_epochs":   1,
    "keep_last_n_ckpts":     3,           # older checkpoints are deleted

    # reproducibility / efficiency
    "seed":                  42,
    "num_workers":           4,
    "fp16":                  False,       # mixed-precision training
    "sample_rate":           16_000,

    # logging
    "log_every_n_steps":     50,
    "run_name":              None,        # optional label shown in log lines

    # wandb -- all optional; set use_wandb=false to disable entirely
    "use_wandb":             True,
    "wandb_project":         "style-prompt-gen",
    "wandb_entity":          "jdm8943-rochester-institute-of-technology",        # your W&B username or team; None uses default
}

REQUIRED = {"h5_path", "meta_path"}

VALID = {
    "d_model":          {192, 384, 768},
    "dialogue_pooler":  {"attentive", "last"},
    "lr_schedule":      {"cosine", "linear", "constant"},
    "num_turns":        set(range(0, 6)),   # 0-5 inclusive
    "num_prefix_tokens": {10, 20, 40},
    "batch_size":       {4, 8, 16, 32},
}


# Config loading + validation

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        user = json.load(f)

    cfg = {**DEFAULTS, **user}

    # check required fields
    for key in REQUIRED:
        if cfg[key] is None:
            raise ValueError(f"Config is missing required field: '{key}'")

    # check enum fields -- only when the value is one we explicitly constrain
    for key, allowed in VALID.items():
        if key in cfg and cfg[key] not in allowed:
            raise ValueError(
                f"Config field '{key}' = {cfg[key]!r} is not one of {sorted(allowed)}"
            )

    # d_model must be divisible by 3 (splits into 3 sub-streams in SCFA)
    if cfg["d_model"] % 3 != 0:
        raise ValueError(f"d_model must be divisible by 3, got {cfg['d_model']}")

    # nhead must divide both d_model and d_model // 3
    d_sub = cfg["d_model"] // 3 # d_sub is dimensionality for the context transformer and 2 speaker transformers
    if cfg["d_model"] % cfg["nhead"] != 0 or d_sub % cfg["nhead"] != 0:
        raise ValueError(
            f"nhead={cfg['nhead']} must divide d_model={cfg['d_model']} "
            f"AND d_model//3={d_sub}"
        )

    return cfg


# Reproducibility

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Model construction

def _unfreeze_top_n_layers(model: nn.Module, layer_attr: str, n: int):
    """Freeze everything first, then selectively unfreeze the top n encoder layers."""
    for p in model.parameters():
        p.requires_grad = False

    if n <= 0:
        return

    layers = getattr(model, layer_attr, None)
    if layers is None:
        log.warning(f"Could not find attribute '{layer_attr}' on {type(model).__name__} -- skipping unfreeze")
        return

    # unfreeze from the top (last layers first, which are most task-relevant)
    for layer in layers[-n:]:
        for p in layer.parameters():
            p.requires_grad = True


def build_text_encoder(cfg: Dict[str, Any], device: torch.device):
    """Load BERT, freeze all layers, then optionally unfreeze the top N."""
    from transformers import AutoModel, AutoTokenizer

    BERT_REPO = "google-bert/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(BERT_REPO)
    model = AutoModel.from_pretrained(BERT_REPO).to(device)

    _unfreeze_top_n_layers(model, "encoder.layer", cfg["num_unfrozen_bert"])

    n = cfg["num_unfrozen_bert"]
    log.info(f"BERT: {n} encoder layer(s) unfrozen")
    return model, tokenizer


def build_audio_encoder(cfg: Dict[str, Any], device: torch.device):
    """Load WavLM-base-plus, freeze all layers, then optionally unfreeze the top N."""
    from transformers import WavLMModel, AutoFeatureExtractor

    WAVLM_REPO = "microsoft/wavlm-base-plus"
    processor = AutoFeatureExtractor.from_pretrained(WAVLM_REPO)
    model = WavLMModel.from_pretrained(WAVLM_REPO).to(device)

    _unfreeze_top_n_layers(model, "encoder.layers", cfg["num_unfrozen_wavlm"])

    n = cfg["num_unfrozen_wavlm"]
    log.info(f"WavLM: {n} encoder layer(s) unfrozen")
    return model, processor


def build_model(cfg: Dict[str, Any], device: torch.device) -> SCFAWithStyleHead:
    log.info("Building model...")

    text_backbone, tokenizer = build_text_encoder(cfg, device)
    audio_backbone, processor = build_audio_encoder(cfg, device)


    embedder = DualModalityEmbedder(
        text_encoder_model_pretrained=text_backbone,
        audio_encoder_model_pretrained=audio_backbone,
        tokenizer=tokenizer,
        processor=processor,
        SAMPLE_RATE=cfg["sample_rate"],
    )

    scfa = SCFA(
        max_turns=cfg["num_turns"],
        embedder=embedder,
        d_model=cfg["d_model"],
        num_ctx_layers=cfg["num_ctx_layers"],
        num_spk_layers=cfg["num_spk_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        nhead=cfg["nhead"],
        dropout=cfg["dropout"],
    ).to(device)

    # 4 * d_model because SCFA cats [z_audio, z_text, z_audio_fused, z_text_fused]
    pooler = DialoguePooler(
        d_model=cfg["d_model"] * 4,
        mode=cfg["dialogue_pooler"],
    ).to(device)

    # TinyLlama embedding dim is fixed at TINYLLAMA_DIM (2048)
    head = StylePromptHead(
        d_model=cfg["d_model"],
        num_prefix_tokens=cfg["num_prefix_tokens"],
        num_mapping_layers=cfg["num_mapping_layers"],
        nhead=cfg["mapping_nhead"],
        dropout=cfg["dropout"],
    ).to(device)

    tokenizer_llm = AutoTokenizer.from_pretrained(TINYLLAMA_REPO)
    llm = AutoModelForCausalLM.from_pretrained(TINYLLAMA_REPO).to(device)
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token

    generator = StylePromptGenerator(
        style_head=head,
        tokenizer=tokenizer_llm,
        llm=llm,
        system_prompt=cfg["system_prompt"],
        max_new_tokens=cfg["max_new_tokens"],
        max_prompt_tokens=cfg["max_prompt_tokens"],
    ).to(device)

    model = SCFAWithStyleHead(scfa=scfa, pooler=pooler, style_generator=generator)
    return model


# Data

def build_dataloaders(cfg: Dict[str, Any]):
    # num_turns == 0 means script-only (text only), represented as 1 turn with no audio
    effective_turns = max(cfg["num_turns"], 1)

    dataset = ConvoStyleDataset(
        h5_path=cfg["h5_path"],
        meta_path=cfg["meta_path"],
        meta_columns=["transcription", "text_description"],  # text_description is the training target
        sample_rate=cfg["sample_rate"],
        num_turns=effective_turns,
        max_len_sec=cfg["max_len_sec"],
    )

    val_size = max(1, int(len(dataset) * cfg["val_split"]))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    loader_kwargs = dict(
        collate_fn=collate_pad,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    # Don't shuffle, rely on seed randomness to ensure reproducible experiments
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs
    )

    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs
    )

    log.info(f"Dataset: {len(train_ds)} train  |  {len(val_ds)} val")
    return train_loader, val_loader, dataset



# Optimizer + scheduler

def build_optimizer_and_scheduler(model: SCFAWithStyleHead, cfg: Dict[str, Any], total_steps: int):
    # only optimize parameters that require gradients
    # (LLM is frozen inside StylePromptGenerator; backbone layers may be partially frozen)
    trainable = [p for p in model.parameters() if p.requires_grad]
    log.info(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    warmup_steps = int(total_steps * cfg["warmup_ratio"])

    if cfg["lr_schedule"] == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    elif cfg["lr_schedule"] == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:  # constant
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, warmup_steps)

    return optimizer, scheduler


# Loss

def compute_loss(
    model: SCFAWithStyleHead,
    batch: Dict[str, Any],
    device: torch.device,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    Teacher-forced cross-entropy over the style prompt tokens.

    The LLM is frozen, so we only flow gradients through the mapping network
    (StylePromptHead), SCFA model, and optionally the unfrozen layers of BERT and WavLM.
    """
    audio      = batch["audio"].to(device)        # (B, T, samples)
    lengths    = batch["lengths"].to(device)      # (B, T)
    text_only  = batch["text_only"].to(device)    # (B, T)
    texts      = batch["transcription"]           # list[B][T]
    speaker_ids = batch["speaker_id"]             # list[B][T]
    targets    = batch["text_description"]            # list[B][T] -- we want the anchor turn's prompt

    # anchor is always the last turn
    anchor_prompts = [chain[-1] for chain in targets]  # list[B]

    # num_turns == 0: treat as text-only by zeroing audio
    if cfg["num_turns"] == 0:
        audio = torch.zeros_like(audio)
        text_only = torch.ones_like(text_only)


    #
    #   RUNNING MODEL
    #

    # Encode turns with context + speaker information
    dialogue_ctx = model.scfa(audio, lengths, texts, speaker_ids, text_only)
    # pool embeddings
    dialogue_vec = model.pooler(dialogue_ctx)
    # get prefix embeddings from the style head
    prefix_embeds = model.style_generator.style_head(dialogue_vec)  # (B, K, TINYLLAMA_DIM)

    B = prefix_embeds.shape[0]
    K = prefix_embeds.shape[1]

    # tokenize the target style descriptions
    tokenizer = model.style_generator.tokenizer
    llm       = model.style_generator.llm

    target_tokens = tokenizer(
        anchor_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg["max_style_desc_tokens"],
    )
    input_ids = target_tokens.input_ids.to(device)   # (B, L)
    attn_mask = target_tokens.attention_mask.to(device)

    # embed target tokens for teacher forcing
    token_embeds = llm.get_input_embeddings()(input_ids)  # (B, L, TINYLLAMA_DIM)

    # optionally prepend system prompt, skip if empty string
    if model.style_generator.system_prompt:
        prompt_tokens = tokenizer(
            [model.style_generator.system_prompt] * B,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg["max_style_desc_tokens"],
        )
        prompt_embeds = llm.get_input_embeddings()(prompt_tokens.input_ids.to(device))

        input_embeds = torch.cat([prefix_embeds, prompt_embeds, token_embeds], dim=1)
        prefix_mask  = torch.ones(B, K + prompt_embeds.shape[1], dtype=torch.long, device=device)
        full_mask    = torch.cat([prefix_mask, attn_mask], dim=1)
    else:
        input_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        prefix_mask  = torch.ones(B, K, dtype=torch.long, device=device)
        full_mask    = torch.cat([prefix_mask, attn_mask], dim=1)

    # build labels: -100 for prefix positions (not supervised), token ids elsewhere
    prefix_labels = torch.full((B, input_embeds.shape[1] - input_ids.shape[1]), -100, device=device)
    token_labels  = input_ids.masked_fill(attn_mask == 0, -100)
    labels        = torch.cat([prefix_labels, token_labels], dim=1)

    # cast input embeddings to TinyLlama dtype bf16
    input_embeds = input_embeds.to(llm.dtype)

    outputs = llm(
        inputs_embeds=input_embeds,
        attention_mask=full_mask,
        labels=labels,
    )

    return outputs.loss



# Checkpoint helpers

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, cfg, out_dir: Path):
    ckpt = {
        "epoch":     epoch,
        "step":      step,
        "loss":      loss,
        "cfg":       cfg,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    path = out_dir / f"ckpt_epoch{epoch:04d}_step{step:07d}.pt"
    torch.save(ckpt, path)
    log.info(f"Saved checkpoint: {path}")
    return path


def prune_old_checkpoints(out_dir: Path, keep: int):
    ckpts = sorted(out_dir.glob("ckpt_epoch*.pt"))
    for old in ckpts[:-keep]:
        old.unlink()
        log.info(f"Removed old checkpoint: {old}")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    log.info(f"Resumed from checkpoint: {path}  (epoch {ckpt['epoch']}, step {ckpt['step']})")
    return ckpt["epoch"], ckpt["step"]


# Training loop

def _grad_norm(model: nn.Module) -> float:
    """Compute global L2 norm of all gradients. Useful for diagnosing training stability."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5


def run_epoch(
    model, loader, optimizer, scheduler, scaler,
    device, cfg, epoch, global_step, wandb_run, is_train=True,
) -> tuple[float, int]:
    model.train(is_train)
    total_loss = 0.0
    n_batches  = 0
    tag = "TRAIN" if is_train else "VAL"
 
    ctx = torch.enable_grad if is_train else torch.no_grad
 
    with ctx():
        for batch in loader:
            with torch.autocast(device_type=device.type, enabled=cfg["fp16"]):
                loss = compute_loss(model, batch, device, cfg)
 
            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if cfg["grad_clip"]:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            cfg["grad_clip"],
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if cfg["grad_clip"]:
                        nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            cfg["grad_clip"],
                        )
                    optimizer.step()
 
                scheduler.step()
                global_step += 1
 
                if global_step % cfg["log_every_n_steps"] == 0:
                    lr = scheduler.get_last_lr()[0]
                    grad_norm = _grad_norm(model)
                    run = f"[{cfg['run_name']}] " if cfg["run_name"] else ""
                    log.info(
                        f"{run}epoch {epoch}  step {global_step}  "
                        f"loss {loss.item():.4f}  lr {lr:.2e}  grad_norm {grad_norm:.3f}"
                    )
                    # step-level metrics -- logged at every log_every_n_steps
                    wandb_log({
                        "train/loss":      loss.item(),
                        "train/lr":        lr,
                        "train/grad_norm": grad_norm,
                    }, step=global_step, run=wandb_run)
 
            total_loss += loss.item()
            n_batches  += 1
 
    avg = total_loss / max(n_batches, 1)
    log.info(f"{tag} epoch {epoch} avg loss: {avg:.4f}")
    return avg, global_step


def train(cfg: Dict[str, Any]):
    set_seed(cfg["seed"])
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
 
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
 
    # save the resolved config for reproducibility
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
 
    wandb_run = wandb_init(cfg)
 
    train_loader, val_loader, dataset = build_dataloaders(cfg)
    model = build_model(cfg, device)
 
    total_steps = len(train_loader) * cfg["num_epochs"]
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, total_steps)
 
    # W&B can watch gradients and parameter histograms if you want them
    # log_freq controls how often histograms are computed -- expensive so keep it low
    if wandb_run is not None:
        import wandb
        wandb_run.watch(model, log="gradients", log_freq=cfg["log_every_n_steps"] * 5)
 
    # mixed-precision scaler; only active when fp16=True
    scaler = torch.amp.GradScaler(device=device) if cfg["fp16"] and device.type == "cuda" else None
 
    start_epoch = 0
    global_step = 0
 
    # resume if a checkpoint exists in the output dir
    existing = sorted(out_dir.glob("ckpt_epoch*.pt"))
    if existing:
        start_epoch, global_step = load_checkpoint(
            str(existing[-1]), model, optimizer, scheduler
        )
        start_epoch += 1  # resume from the next epoch
 
    best_val_loss = float("inf")
 
    for epoch in range(start_epoch, cfg["num_epochs"]):
        train_loss, global_step = run_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, cfg, epoch, global_step, wandb_run, is_train=True,
        )
 
        # epoch-level train loss (separate key from step-level so the chart is clean)
        wandb_log({"epoch/train_loss": train_loss, "epoch": epoch}, step=global_step, run=wandb_run)
 
        if (epoch + 1) % cfg["eval_every_n_epochs"] == 0:
            val_loss, _ = run_epoch(
                model, val_loader, optimizer, scheduler, scaler,
                device, cfg, epoch, global_step, wandb_run, is_train=False,
            )
            wandb_log({"epoch/val_loss": val_loss, "epoch": epoch}, step=global_step, run=wandb_run)
 
            # track best val loss so you can spot divergence in the W&B dashboard
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wandb_log({"epoch/best_val_loss": best_val_loss}, step=global_step, run=wandb_run)
 
        if (epoch + 1) % cfg["save_every_n_epochs"] == 0:
            ckpt_path = save_checkpoint(
                model, optimizer, scheduler, epoch, global_step, train_loss, cfg, out_dir
            )
            prune_old_checkpoints(out_dir, cfg["keep_last_n_ckpts"])
 
            # log the checkpoint as a W&B artifact so you can restore any saved version
            if wandb_run is not None:
                import wandb
                artifact = wandb.Artifact(
                    name=f"checkpoint-{wandb_run.id}",
                    type="model",
                    metadata={"epoch": epoch, "step": global_step, "val_loss": best_val_loss if best_val_loss != float("inf") else None},

                )
                artifact.add_file(str(ckpt_path))
                wandb_run.log_artifact(artifact)
 
    log.info("Training complete.")
    wandb_finish(wandb_run)


# W&B helpers
 
def wandb_init(cfg: Dict[str, Any]):
    if not cfg.get("use_wandb", True):
        return None

    try:
        import wandb
    except ImportError:
        log.warning("wandb not installed -- skipping W&B tracking.")
        return None

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        log.warning("WANDB_API_KEY not set -- skipping W&B tracking.")
        return None

    # explicitly authenticate so wandb never falls through to the interactive prompt
    wandb.login(key=api_key, relogin=False)

    try:
        run = wandb.init(
            project=cfg["wandb_project"],
            entity=cfg.get("wandb_entity"),
            name=cfg.get("run_name"),
            config=cfg,
            resume="allow",
        )
        log.info(f"W&B run: {run.url}")
        return run
    except Exception as e:
        log.warning(f"W&B init failed ({e}) -- continuing without tracking.")
        return None
 
 
def wandb_log(metrics: dict, step: int, run):
    """Log a dict of metrics. No-ops if W&B is disabled."""
    if run is None:
        return
    run.log(metrics, step=step)
 
 
def wandb_finish(run):
    if run is not None:
        run.finish()


# Entry point

def parse_args():
    p = argparse.ArgumentParser(description="Train SCFAWithStyleHead from a JSON config.")
    p.add_argument("--config", required=True, help="Path to hyperparameter config JSON")
    # allow inline overrides: --override learning_rate=1e-4 batch_size=16
    p.add_argument(
        "--override", nargs="*", metavar="KEY=VALUE",
        help="Override individual config fields (e.g. --override learning_rate=1e-4)"
    )
    return p.parse_args()


def apply_overrides(cfg: Dict[str, Any], overrides):
    if not overrides:
        return cfg
    for item in overrides:
        key, _, raw = item.partition("=")
        if key not in cfg:
            raise ValueError(f"Unknown config key in override: '{key}'")
        # cast to the same type as the default
        default_val = DEFAULTS.get(key)
        if default_val is None:
            cfg[key] = raw  # can't infer type; keep as string
        elif isinstance(default_val, bool):
            cfg[key] = raw.lower() in ("1", "true", "yes")
        elif isinstance(default_val, int):
            cfg[key] = int(raw)
        elif isinstance(default_val, float):
            cfg[key] = float(raw)
        else:
            cfg[key] = raw
        log.info(f"Override: {key} = {cfg[key]!r}")
    return cfg


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)
    train(cfg)
