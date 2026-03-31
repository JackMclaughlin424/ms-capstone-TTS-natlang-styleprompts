"""
Baseline: few-shot style prompt generation using TinyLlama via HuggingFace transformers.
Samples 3 random chains from ConvoStyleDataset to use as in-context examples,
then predicts the style for a query chain's final utterance.
"""

import random
import textwrap
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ConvoStyleDataset import ConvoStyleDataset


# same repo used by StylePromptGenerator -- keep these in sync
TINYLLAMA_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 80

H5_PATH   = "../data_TEMP/merged_audio_full.h5"
META_PATH = "../data_TEMP/merged_metadata_full.parquet"

NUM_TURNS      = 5   # chain length
NUM_FEW_SHOT   = 3   # examples in the system prompt
RANDOM_SEED    = 42


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
        You are a speech style annotation assistant for a conversational text-to-speech system.
        Your task is to predict an appropriate natural-language style description for the NEXT utterance
        to be synthesized, given the dialogue history and the script (transcription) of that next utterance.

        The style description should reflect how the speech should sound -- covering aspects like emotion,
        speaking rate, vocal quality, and energy level -- and should be consistent with the speaker's
        established style and the conversational flow. Be concise: 1-3 sentences is ideal.

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

    # sample NUM_FEW_SHOT + 1 distinct indices: first 3 are examples, last is the query
    sampled_indices = random.sample(range(len(ds)), NUM_FEW_SHOT + 1)
    few_shot_indices = sampled_indices[:NUM_FEW_SHOT]
    query_index      = sampled_indices[-1]

    few_shot_chains = [ds[i] for i in few_shot_indices]
    query_chain     = ds[query_index]

    system_prompt = build_system_prompt(few_shot_chains)
    user_prompt   = build_user_prompt(query_chain)

    print("\n" + "="*60)
    print("SYSTEM PROMPT")
    print("="*60)
    print(system_prompt)

    print("\n" + "="*60)
    print("USER PROMPT (query chain)")
    print("="*60)
    print(user_prompt)

    print("\n" + "="*60)
    print(f"Loading TinyLlama on {device}...")
    print("="*60)
    tokenizer, model = load_tinyllama(device)

    prediction = query_tinyllama(tokenizer, model, system_prompt, user_prompt, device)

    ground_truth = query_chain[-1].get("text_description", "N/A").strip()

    print(f"\nPredicted style : {prediction}")
    print(f"Ground truth    : {ground_truth}")


if __name__ == "__main__":
    main()