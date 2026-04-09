"""
Baseline: few-shot style prompt generation using TinyLlama via HuggingFace transformers.
Samples 3 random chains from ConvoStyleDataset to use as in-context examples,
then predicts the style for a query chain's final utterance.
"""

import os
import json
import random
import gc
import textwrap
import argparse
import numpy as np
import pandas as pd 

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ConvoStyleDataset import ConvoStyleDataset

from train_helpers import compute_bertscore, compute_meteor

# same repo used by StylePromptGenerator -- keep these in sync
LLM_REPO = "meta-llama/Llama-3.2-3B-Instruct"  # or whichever you choose
LLM_DIM  = 3072  # must match model's hidden_size



# W&B config -- set USE_WANDB=False or unset WANDB_API_KEY to disable
USE_WANDB      = True
WANDB_PROJECT  = "style-prompt-gen"
WANDB_ENTITY   = "jdm8943-rochester-institute-of-technology"
WANDB_RUN_NAME = "baseline-llama3B-fewshot"


def load_dataset(h5_path: str, meta_path: str, num_turns: int, max_len_sec: int) -> ConvoStyleDataset:
    return ConvoStyleDataset(
        h5_path=h5_path,
        meta_path=meta_path,
        meta_columns=["transcription", "text_description", "speakerid", "conv_id"],
        num_turns=num_turns,
        max_len_sec=max_len_sec,
    )



def chain_to_text(chain: list) -> str:
    lines = []
    for i, utt in enumerate(chain):
        text    = utt.get("transcription", "").strip()
        is_last = i == len(chain) - 1

        lines.append(f"[Turn {i+1}]")
        lines.append(f"  Transcription: \"{text}\"")
        if is_last:
            lines.append(f"  Speaking style:")
        else:
            style = (utt.get("text_description") or "unknown").strip()
            lines.append(f"  Speaking style: {style}")

    return "\n".join(lines)






def build_few_shot_example(chain: list) -> str:
    lines = []
    for i, utt in enumerate(chain):
        text  = utt.get("transcription", "").strip()
        style = (utt.get("text_description") or "unknown").strip()

        lines.append(f"[Turn {i+1}]")
        lines.append(f"  Transcription: \"{text}\"")
        lines.append(f"  Speaking style: {style}")

    return "\n".join(lines)





def build_system_prompt(few_shot_chains: list[list]) -> str:
    task_description = textwrap.dedent("""\
        You are a prompt assistant for a conversational, instruction-guided text-to-speech system.
        Your task is to predict an appropriate natural-language style description (style prompt) for the NEXT utterance
        to be synthesized, given the dialogue history and the script (transcription) of that next utterance.

        The style description should reflect how the speech should sound -- covering aspects like emotion,
        speaking rate, vocal quality, and energy level -- and should be consistent with the speaker's
        established style, and be appropriate to the conversational flow. Be concise: 1-3 sentences is ideal.

        Each turn shows the transcription, and (for all turns except the final query turn) the speaking style.
        For the final turn, only the transcription is provided.
        Predict the style for that final turn and output it as "Style: <your prediction>".
        Respond with only the content of your style prediction for the final turn, nothing else.

    """)


    examples_block = "\n\n---\n\n".join(
        f"Example {i+1}:\n{build_few_shot_example(chain)}"
        for i, chain in enumerate(few_shot_chains)
    )

    return f"{task_description}\nHere are {len(few_shot_chains)} examples:\n\n{examples_block}"


def build_user_prompt(query_chain: list) -> str:
    dialogue_block = chain_to_text(query_chain)
    return (
        "Given the dialogue history and the script for the final turn, "
        "predict only the final style description.\n\n"
        f"{dialogue_block}\n"
    )




# wandb helpers

def wandb_init(cfg: dict):
    if not USE_WANDB:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed -- skipping W&B tracking.")
        return None
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("WANDB_API_KEY not set -- skipping W&B tracking.")
        return None
    wandb.login(key=api_key, relogin=False)
    try:
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=WANDB_RUN_NAME,
            config=cfg,
        )
        print(f"W&B run: {run.url}")
        return run
    except Exception as e:
        print(f"W&B init failed ({e}) -- continuing without tracking.")
        return None


def wandb_log(metrics: dict, run):
    if run is not None:
        run.log(metrics)


def wandb_finish(run):
    if run is not None:
        run.finish()




def load_llm(device: str, repo: str = LLM_REPO):
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16)
    model = model.to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # causal LMs must left-pad for batched generation
    return tokenizer, model




def batch_query_llm(
    tokenizer, model, full_prompts: list[str], device: str,
    max_new_tokens: int, batch_size: int = 8
) -> list[str]:
    results = []
    for start in range(0, len(full_prompts), batch_size):
        batch = full_prompts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        for seq in output_ids:
            generated = seq[prompt_len:]
            results.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
        
        del inputs, output_ids
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return results




def main(
    h5_path: str,
    meta_path: str,
    num_turns: int = 5,
    num_few_shot: int = 3,
    num_eval_samples: int = 50,
    max_new_tokens: int = 80,
    inference_batch_size: int = 8,
    max_len_sec: int = 15,
    seed: int = 42,
    llm_repo: str = LLM_REPO,
    output_path: str = "baseline_outputs.json",
):


    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    ds = load_dataset(h5_path, meta_path, num_turns, max_len_sec)
    print(f"  {len(ds)} chains available")


    # reproduce the same 10% held-out test split as sweep.py (seeds must be same)
    all_conv_ids = np.array(pd.read_parquet(meta_path)["conv_id"].unique())
    rng_split = np.random.default_rng(seed)
    shuffled_all = all_conv_ids.copy()
    rng_split.shuffle(shuffled_all)
    n_test   = max(1, int(len(shuffled_all) * 0.10))
    test_ids = set(shuffled_all[:n_test])
    print(f"  Test split: {len(test_ids)} conv_ids (10% held-out, seed={seed})")

    def has_valid_styles(chain):
        return all(
            chain_row.get("text_description") not in (None, "")
            and not (isinstance(chain_row.get("text_description"), float) and np.isnan(chain_row.get("text_description")))
            for chain_row in chain
        )

    # eval pool: test split only
    test_indices = [
        i for i, chain in enumerate(ds._chains)
        if chain[-1].get("conv_id") in test_ids and has_valid_styles(chain)
    ]

    # few-shot pool: train split only, so no eval chain leaks into context
    train_indices = [
        i for i, chain in enumerate(ds._chains)
        if chain[-1].get("conv_id") not in test_ids and has_valid_styles(chain)
    ]

    query_indices = random.sample(test_indices, min(num_eval_samples, len(test_indices)))
    print(f"\n{len(query_indices)} query chains sampled for evaluation.")
    print(f"  Few-shot pool size: {len(train_indices)} train chains")

    print(f"\nLoading LLM on {device}...")
    tokenizer, model = load_llm(device, repo=llm_repo)

    wandb_run = wandb_init({
        "model":            llm_repo,
        "num_turns":        num_turns,
        "num_few_shot":     num_few_shot,
        "num_eval_samples": num_eval_samples,
        "max_new_tokens":   max_new_tokens,
        "max_len_sec":      max_len_sec,
        "seed":             seed,
    })


    all_preds, all_refs = [], []
    records = []


     # build all prompts up front
    full_prompts, ground_truths, meta = [], [], []
    for qi in query_indices:
        few_shot_chains = [ds[i] for i in random.sample(train_indices, num_few_shot)]
        system_prompt   = build_system_prompt(few_shot_chains)
        query_chain     = ds[qi]
        user_prompt     = build_user_prompt(query_chain)
        full_prompts.append(f"{system_prompt}\n\n---\n\n{user_prompt}")
        ground_truths.append(query_chain[-1].get("text_description", "").strip())
        meta.append({"index": qi, "system_prompt": system_prompt, "user_prompt": user_prompt})
        
    # batch inference
    print(f"\nRunning batched inference over {len(full_prompts)} prompts...")
    predictions = batch_query_llm(tokenizer, model, full_prompts, device
                                  , max_new_tokens, batch_size=inference_batch_size)

    # collect results 
    all_preds, all_refs = [], []
    records = []
    for idx, (pred, gt, m) in enumerate(zip(predictions, ground_truths, meta)):
        records.append({**m, "prediction": pred, "ground_truth": gt})
        all_preds.append(pred)
        all_refs.append(gt)
        print(f"\n[{idx+1}/{len(query_indices)}]")
        print(f"  Predicted : {pred}")
        print(f"  Reference : {gt}")


    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} inference records to {output_path}")

    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)

    bs_metrics  = compute_bertscore(all_preds, all_refs, device=device)
    met_metrics = compute_meteor(all_preds, all_refs)

    print(f"BERTScore F1  : {bs_metrics['bertscore_f1_mean']:.4f} (±{bs_metrics['bertscore_f1_std']:.4f})")
    print(f"BERTScore P   : {bs_metrics['bertscore_precision_mean']:.4f} (±{bs_metrics['bertscore_precision_std']:.4f})")
    print(f"BERTScore R   : {bs_metrics['bertscore_recall_mean']:.4f} (±{bs_metrics['bertscore_recall_std']:.4f})")
    print(f"METEOR        : {met_metrics['meteor_mean']:.4f} (±{met_metrics['meteor_std']:.4f})")

    wandb_log({**bs_metrics, **met_metrics}, wandb_run)
    wandb_finish(wandb_run)

    return {**bs_metrics, **met_metrics}


def parse_args():
    p = argparse.ArgumentParser(description="Few-shot LLM baseline for style prompt prediction.")
    p.add_argument("--h5_path",          required=True,              help="Path to merged_audio.h5")
    p.add_argument("--meta_path",        required=True,              help="Path to merged_metadata.parquet")
    p.add_argument("--num_turns",        type=int,   default=5,      help="Conversation chain length (default: 5)")
    p.add_argument("--num_few_shot",     type=int,   default=3,      help="Number of in-context examples (default: 3)")
    p.add_argument("--num_eval_samples", type=int,   default=50,     help="Number of query chains to evaluate (default: 50)")
    p.add_argument("--max_new_tokens",   type=int,   default=80,     help="Max tokens to generate per prediction (default: 80)")
    p.add_argument("--inference_batch_size",   type=int,   default=8,     help="Size of prompt batch to run inference. Adjust for VRAM constraints.")
    p.add_argument("--max_len_sec",      type=int,   default=15,     help="Max audio length in seconds (default: 15)")
    p.add_argument("--llm_repo", type=str, default=LLM_REPO, help="HuggingFace repo for the LLM (default: Llama 3.2 3B)")

    p.add_argument("--seed",             type=int,   default=42,     help="Random seed (default: 42)")
    p.add_argument("--output_path", type=str, default="baseline_outputs.json",
            help="Path to save per-inference JSON output (default: baseline_outputs.json)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
