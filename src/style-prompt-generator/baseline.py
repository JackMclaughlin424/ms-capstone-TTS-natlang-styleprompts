"""
Baseline: few-shot style prompt generation using TinyLlama via HuggingFace transformers.
Samples 3 random chains from ConvoStyleDataset to use as in-context examples,
then predicts the style for a query chain's final utterance.
"""

import os
import random
import textwrap

import nltk
from nltk.translate.meteor_score import meteor_score as _meteor
from bert_score import score as _bert_score
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ConvoStyleDataset import ConvoStyleDataset

from train_helpers import compute_bertscore, compute_meteor

# same repo used by StylePromptGenerator -- keep these in sync
TINYLLAMA_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 80

H5_PATH   = "../data_TEMP/merged_audio_full.h5"
META_PATH = "../data_TEMP/merged_metadata_full.parquet"

NUM_TURNS      = 5   # chain length
NUM_FEW_SHOT   = 3   # examples in the system prompt
NUM_EVAL_SAMPLES = 50  # how many query chains to evaluate
RANDOM_SEED    = 42

# W&B config -- set USE_WANDB=False or unset WANDB_API_KEY to disable
USE_WANDB      = True
WANDB_PROJECT  = "style-prompt-gen"
WANDB_ENTITY   = "jdm8943-rochester-institute-of-technology"
WANDB_RUN_NAME = "baseline-tinyllama-fewshot"


def load_dataset(num_turns: int) -> ConvoStyleDataset:
    return ConvoStyleDataset(
        h5_path=H5_PATH,
        meta_path=META_PATH,
        # audio not needed here -- we only use text fields
        meta_columns=["transcription", "text_description", "speakerid"],
        num_turns=num_turns,
    )


def chain_to_text(chain: list, include_last_style: bool = True) -> str:
    """
    Render a chain as a readable dialogue block.
    The last turn is the "current script" -- its style is what we want to predict,
    so we only include its transcription, not its style.
    """
    lines = []
    for i, utt in enumerate(chain):
        speaker = utt.get("speakerid", f"Speaker{i}")
        text    = utt.get("transcription", "").strip()
        style   = utt.get("text_description", "").strip()
        is_last = i == len(chain) - 1

        if is_last:
            # this is the turn we're predicting -- no style label
            lines.append(f"[Turn {i+1}] {speaker}: \"{text}\"")
            lines.append(f"  -> Style: ???")
        else:
            style_str = style if style else "unknown"
            lines.append(f"[Turn {i+1}] {speaker}: \"{text}\"")
            lines.append(f"  -> Style: {style_str}")

    return "\n".join(lines)


def build_few_shot_example(chain: list) -> str:
    """
    Renders one complete example (all turns including the last turn's style)
    for use inside the system prompt.
    """
    lines = []
    for i, utt in enumerate(chain):
        speaker = utt.get("speakerid", f"Speaker{i}")
        text    = utt.get("transcription", "").strip()
        style   = utt.get("text_description", "").strip()
        is_last = i == len(chain) - 1

        style_str = style if style else "unknown"
        lines.append(f"[Turn {i+1}] {speaker}: \"{text}\"")

        if is_last:
            # the answer we want the model to reproduce in real queries
            lines.append(f"  -> Style: {style_str}")
        else:
            lines.append(f"  -> Style: {style_str}")

    return "\n".join(lines)


def build_system_prompt(few_shot_chains: list[list]) -> str:
    task_description = textwrap.dedent("""\
        You are a prompt assistant for a conversational, instruction-guided text-to-speech system.
        Your task is to predict an appropriate natural-language style description (style prompt) for the NEXT utterance
        to be synthesized, given the dialogue history and the script (transcription) of that next utterance.

        The style description should reflect how the speech should sound -- covering aspects like emotion,
        speaking rate, vocal quality, and energy level -- and should be consistent with the speaker's
        established style, and be appropriate to the conversational flow. Be concise: 1-3 sentences is ideal.

        Each turn shows the speaker, their transcription, and the style used for that turn.
        For the final turn, the style is marked as "???" -- that is what you must predict.
        Respond with only the style description for the final turn, nothing else.
    """)

    examples_block = "\n\n---\n\n".join(
        f"Example {i+1}:\n{build_few_shot_example(chain)}"
        for i, chain in enumerate(few_shot_chains)
    )

    return f"{task_description}\nHere are {len(few_shot_chains)} examples:\n\n{examples_block}"


def build_user_prompt(query_chain: list) -> str:
    dialogue_block = chain_to_text(query_chain, include_last_style=False)
    return (
        "Given the dialogue history, the styles used in each turn, and the current script "
        "(the final turn marked ???), predict an appropriate style for the speech to be synthesized.\n\n"
        f"{dialogue_block}\n\n"
        "Predicted style for the final turn:"
    )


# wandb helpers

def wandb_init():
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
            config={
                "model":            TINYLLAMA_REPO,
                "num_turns":        NUM_TURNS,
                "num_few_shot":     NUM_FEW_SHOT,
                "num_eval_samples": NUM_EVAL_SAMPLES,
                "max_new_tokens":   MAX_NEW_TOKENS,
                "random_seed":      RANDOM_SEED,
            },
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




def load_tinyllama(device: str):
    tokenizer = AutoTokenizer.from_pretrained(TINYLLAMA_REPO)
    model = AutoModelForCausalLM.from_pretrained(TINYLLAMA_REPO, torch_dtype=torch.float32)
    model = model.to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def query_tinyllama(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    system_prompt: str,
    user_prompt: str,
    device: str,
) -> str:
    # apply_chat_template handles the [INST] / <<SYS>> formatting for us
    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # appends the assistant turn opener
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # slice off the input tokens so we only decode the generated part
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    random.seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    ds = load_dataset(NUM_TURNS)
    print(f"  {len(ds)} chains available")

    # Reserve NUM_FEW_SHOT indices as fixed in-context examples;
    # sample NUM_EVAL_SAMPLES distinct query indices from the remainder.
    all_indices = list(range(len(ds)))
    few_shot_indices = random.sample(all_indices, NUM_FEW_SHOT)
    remaining = [i for i in all_indices if i not in set(few_shot_indices)]
    query_indices = random.sample(remaining, min(NUM_EVAL_SAMPLES, len(remaining)))

    few_shot_chains = [ds[i] for i in few_shot_indices]
    system_prompt   = build_system_prompt(few_shot_chains)

    print("\n" + "="*60)
    print("SYSTEM PROMPT")
    print("="*60)
    print(system_prompt)

    print(f"\n{len(query_indices)} query chains sampled for evaluation.")
    print(f"\nLoading TinyLlama on {device}...")
    tokenizer, model = load_tinyllama(device)

    wandb_run = wandb_init()

    all_preds, all_refs = [], []

    for idx, qi in enumerate(query_indices):
        query_chain  = ds[qi]
        user_prompt  = build_user_prompt(query_chain)
        prediction   = query_tinyllama(tokenizer, model, system_prompt, user_prompt, device)
        ground_truth = query_chain[-1].get("text_description", "").strip()

        all_preds.append(prediction)
        all_refs.append(ground_truth)

        print(f"\n[{idx+1}/{len(query_indices)}]")
        print(f"  Predicted : {prediction}")
        print(f"  Reference : {ground_truth}")

    # Compute and report evaluation metrics
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



if __name__ == "__main__":
    main()