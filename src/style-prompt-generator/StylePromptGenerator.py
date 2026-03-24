import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class StylePromptHead(nn.Module):
    """
    Prefix-token based style prompt generation head. Based on StyleCap (Yamauichi et al. 2024)

    Takes the concatenated dialogue context from SCFA (z_audio + z_text + 
    z_audio_fused + z_text_fused vector) and generates soft prefix tokens that
    are prepended to the LLM's input sequence to steer style prompt generation.

    The prefix tokens act as a "virtual prompt" -- the LLM never sees raw dialogue
    embeddings, just these learned continuous tokens that encode conversational style
    context.
    """

    def __init__(self, d_model, d_llm, num_prefix_tokens, num_llm_layers, dropout=0.1):
        super().__init__()

        self.num_prefix_tokens = num_prefix_tokens
        self.num_llm_layers    = num_llm_layers
        self.d_llm             = d_llm

        # compress each stream independently before fusing
        # -> keeps gradients cleaner
        # stream specializes before merge
        self.stream_compressors = nn.ModuleList([
            nn.Linear(d_model, d_model // 2) for _ in range(4)
        ])

        # fuse compressed streams
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_llm),  # 4 * (d_model // 2) = 2 * d_model
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        self.prefix_proj = nn.Linear(d_llm, num_prefix_tokens * num_llm_layers * 2 * d_llm)
        self.norm = nn.LayerNorm(d_llm)

    def forward(self, dialogue_ctx, turn_idx=None):
        # dialogue_ctx: (B, T, 4 * d_model)
        B, T, D = dialogue_ctx.shape
        d_model = D // 4

        if turn_idx is None:
            ctx = dialogue_ctx[:, -1, :]
        else:
            idx = turn_idx.view(B, 1, 1).expand(B, 1, D)
            ctx = dialogue_ctx.gather(1, idx).squeeze(1)

        # split back into 4 streams and compress each independently
        streams = ctx.split(d_model, dim=-1)  # 4 x (B, d_model)
        compressed = [F.relu(comp(s)) for comp, s in zip(self.stream_compressors, streams)]

        h = self.fusion(torch.cat(compressed, dim=-1))  # (B, d_llm)
        h = self.norm(h)

        prefix_flat = self.prefix_proj(h)
        prefix = prefix_flat.view(B, self.num_llm_layers, 2, self.num_prefix_tokens, self.d_llm)

        past_key_values = [
            (prefix[:, i, 0], prefix[:, i, 1]) for i in range(self.num_llm_layers)
        ]
        return past_key_values


class SCFAWithStyleHead(nn.Module):
    """
    Wraps SCFA + StylePromptHead into one forward pass.
    """

    def __init__(self, scfa: "SCFA", style_head: StylePromptHead):
        super().__init__()
        self.scfa       = scfa
        self.style_head = style_head

    def forward(
        self,
        audio:       torch.Tensor,
        lengths:     torch.Tensor,
        texts,
        speaker_ids,
        text_only:   Optional[torch.Tensor] = None,
        turn_idx:    Optional[torch.Tensor] = None,
    ):
        # run dialogue encoder to get per-turn contextual embeddings
        text_emb, audio_emb = self.scfa.embedder(audio, lengths, texts, text_only)

        z_audio = self.scfa._intra_modal_encode(
            audio_emb, self.scfa.ctx_audio, self.scfa.spk_audio,
            self.scfa.intra_gate_audio, speaker_ids
        )
        z_text = self.scfa._intra_modal_encode(
            text_emb, self.scfa.ctx_text, self.scfa.spk_text,
            self.scfa.intra_gate_text, speaker_ids
        )

        z_audio_fused, z_text_fused = self.scfa.cfa(z_audio, z_text)

        # concatenate all four streams -- same as SCFA classifier input
        combined = torch.cat([z_audio, z_text, z_audio_fused, z_text_fused], dim=-1)
        # (B, T, 4 * d_model)

        past_key_values = self.style_head(combined, turn_idx)

        return past_key_values, combined