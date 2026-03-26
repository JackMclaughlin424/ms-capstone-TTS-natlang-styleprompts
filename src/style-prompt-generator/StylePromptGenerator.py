import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM


TINYLLAMA_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TINYLLAMA_DIM  = 2048  # word embedding dim -- prefix vectors must match this


def load_tinyllama(device: str = "cpu", torch_dtype=torch.float32):
    tokenizer = AutoTokenizer.from_pretrained(TINYLLAMA_REPO)
    model = AutoModelForCausalLM.from_pretrained(TINYLLAMA_REPO, torch_dtype=torch_dtype)
    model = model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


class StylePromptHead(nn.Module):
    """
    Projects SCFA dialogue context into K prefix vectors in TinyLlama word
    embedding space, following ClipCap / StyleCap.

    Pipeline:
        1. Compress + fuse the 4 SCFA streams into a single context vector z
        2. Transformer mapping network: K learnable constants attend to z,
           producing K output vectors in LLM embedding space

    The K output vectors are prepended directly to the LLM input sequence.
    No KV cache injection -- the LLM attends to prefix positions normally.

    Args:
        d_model:            SCFA per-stream dim (typically 768)
        num_prefix_tokens:  K -- number of prefix vectors to produce
        num_mapping_layers: transformer encoder depth in the mapping network
        nhead:              attention heads in the mapping network
        dropout:            dropout throughout
    """

    def __init__(
        self,
        d_model:            int,
        num_prefix_tokens:  int,
        num_mapping_layers: int   = 8,
        nhead:              int   = 8,
        dropout:            float = 0.1,
    ):
        super().__init__()

        self.num_prefix_tokens = num_prefix_tokens

        # stream fusion
        # compress each of the 4 SCFA streams independently before merging
        self.stream_compressors = nn.ModuleList([
            nn.Linear(d_model, d_model // 2) for _ in range(4)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, TINYLLAMA_DIM),  # 4 * (d_model // 2) = 2 * d_model
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(TINYLLAMA_DIM)

        # transformer mapping network:
        # K learnable constants are the queries, z is the memory they attend to
        self.prefix_const = nn.Parameter(torch.randn(num_prefix_tokens, TINYLLAMA_DIM))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TINYLLAMA_DIM,
            nhead=nhead,
            dim_feedforward=TINYLLAMA_DIM * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_mapping_layers)

    def forward(
        self,
        dialogue_ctx: torch.Tensor,              # (B, T, 4 * d_model)
        turn_idx:     Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                           # (B, K, TINYLLAMA_DIM)
        B, T, D = dialogue_ctx.shape
        d_model = D // 4

        # select the target turn -- default to last
        if turn_idx is None:
            ctx = dialogue_ctx[:, -1, :]
        else:
            idx = turn_idx.view(B, 1, 1).expand(B, 1, D)
            ctx = dialogue_ctx.gather(1, idx).squeeze(1)

        # fuse 4 streams -> single context vector z
        streams    = ctx.split(d_model, dim=-1)
        compressed = [F.relu(comp(s)) for comp, s in zip(self.stream_compressors, streams)]
        z = self.norm(self.fusion(torch.cat(compressed, dim=-1)))  # (B, TINYLLAMA_DIM)

        # tile constants across batch, concatenate with z, run transformer
        # z sits at position 0 as memory; constants learn to query it
        consts = self.prefix_const.unsqueeze(0).expand(B, -1, -1)  # (B, K, TINYLLAMA_DIM)
        seq    = torch.cat([z.unsqueeze(1), consts], dim=1)         # (B, 1+K, TINYLLAMA_DIM)
        out    = self.transformer(seq)                               # (B, 1+K, TINYLLAMA_DIM)

        return out[:, 1:, :]  # drop z position, return K prefix vectors


class StylePromptGenerator(nn.Module):
    """
    StylePromptHead + frozen TinyLlama.

    Prefix vectors are prepended to the (optionally present) system prompt
    embeddings, then TinyLlama generates autoregressively. Only the
    StylePromptHead is trained.

    Args:
        style_head:     trained StylePromptHead
        tokenizer:      TinyLlama tokenizer
        llm:            frozen TinyLlama
        system_prompt:  text appended after prefix embeddings.
                        None = prefix-only generation (strict ClipCap style).
        max_new_tokens: generation length cap
    """

    def __init__(
        self,
        style_head:    StylePromptHead,
        tokenizer:     AutoTokenizer,
        llm:           AutoModelForCausalLM,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 80,
    ):
        super().__init__()
        self.style_head     = style_head
        self.tokenizer      = tokenizer
        self.llm            = llm
        self.system_prompt  = system_prompt
        self.max_new_tokens = max_new_tokens

        for p in self.llm.parameters():
            p.requires_grad = False

    def _embed_text(self, texts: List[str], device) -> tuple:
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        input_ids = tokens.input_ids.to(device)
        attn_mask = tokens.attention_mask.to(device)
        embeds    = self.llm.get_input_embeddings()(input_ids)  # (B, L, TINYLLAMA_DIM)
        return embeds, attn_mask

    @torch.no_grad()
    def generate(
        self,
        dialogue_ctx: torch.Tensor,
        turn_idx:     Optional[torch.Tensor] = None,
    ) -> List[str]:
        B      = dialogue_ctx.shape[0]
        device = dialogue_ctx.device

        prefix_embeds = self.style_head(dialogue_ctx, turn_idx)  # (B, K, TINYLLAMA_DIM)
        K             = prefix_embeds.shape[1]
        prefix_mask   = torch.ones(B, K, dtype=torch.long, device=device)

        if self.system_prompt is not None:
            prompt_embeds, prompt_mask = self._embed_text([self.system_prompt] * B, device)
            inputs_embeds  = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            attention_mask = torch.cat([prefix_mask, prompt_mask], dim=1)
        else:
            inputs_embeds  = prefix_embeds
            attention_mask = prefix_mask

        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def forward(
        self,
        dialogue_ctx: torch.Tensor,
        turn_idx:     Optional[torch.Tensor] = None,
    ) -> List[str]:
        return self.generate(dialogue_ctx, turn_idx)


class SCFAWithStyleHead(nn.Module):
    """
    Full pipeline: SCFA dialogue encoder -> StylePromptGenerator -> style prompt strings.
    """
 
    def __init__(self, scfa: "SCFA", style_generator: StylePromptGenerator):
        super().__init__()
        self.scfa            = scfa
        self.style_generator = style_generator
 
    def forward(
        self,
        audio:      torch.Tensor,
        lengths:    torch.Tensor,
        texts,
        speaker_ids,
        text_only:  Optional[torch.Tensor] = None,
        turn_idx:   Optional[torch.Tensor] = None,
    ):
        # SCFA.forward() returns cat([z_audio, z_text, z_audio_fused, z_text_fused], dim=-1)
        dialogue_embedding      = self.scfa(audio, lengths, texts, speaker_ids, text_only)
        style_prompts = self.style_generator(dialogue_embedding, turn_idx)
 
        return style_prompts, dialogue_embedding
 
 
def build_style_generator(
    scfa:               "SCFA",
    d_model:            int           = 768,
    num_prefix_tokens:  int           = 10,
    num_mapping_layers: int           = 8,
    nhead:              int           = 8,
    max_new_tokens:     int           = 80,
    system_prompt:      Optional[str] = None,
    device:             str           = "cpu",
    torch_dtype                        = torch.float32,
) -> SCFAWithStyleHead:
    """
    Loads TinyLlama, wires everything up, returns a ready-to-use SCFAWithStyleHead.
 
    system_prompt controls text conditioning after the prefix:
        None  -> prefix-only generation (ClipCap / StyleCap style)
        str   -> prefix + text anchor
 
    Example:
        model = build_style_generator(scfa, device="cuda")
        style_prompts, ctx = model(audio, lengths, texts, speaker_ids)
    """
    tokenizer, llm = load_tinyllama(device=device, torch_dtype=torch_dtype)
 
    head = StylePromptHead(
        d_model=d_model,
        num_prefix_tokens=num_prefix_tokens,
        num_mapping_layers=num_mapping_layers,
        nhead=nhead,
    )
 
    generator = StylePromptGenerator(
        style_head=head,
        tokenizer=tokenizer,
        llm=llm,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
    )
 
    return SCFAWithStyleHead(scfa=scfa, style_generator=generator)