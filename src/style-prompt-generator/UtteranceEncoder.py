import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentivePooling(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.scorer = nn.Linear(dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # hidden: (B, T, d),  mask: (B, T) -- 1 for real tokens, 0 for padding
        scores = self.scorer(hidden).squeeze(-1)  # (B, T)

        if mask is not None:
            # push padding positions to -inf so softmax zeroes them out
            scores = scores.masked_fill(~mask.bool(), float("-inf"))

        weights = F.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * hidden).sum(dim=1)  # (B, d)
    
class ModalityEncoder(nn.Module):

    def __init__(self, backbone, pooler):
        super().__init__()
        self.backbone = backbone
        self.pooler   = pooler

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        return self.pooler(hidden, mask)  # (B, d)
    

def encode_text_inputs(texts, tokenizer):
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

def encode_audio_inputs(audio, lengths, processor):
    inputs = processor(
        list(audio.cpu().numpy()),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    ).to(device)
    return inputs, lengths


class DualModalityEmbedder(nn.Module):

    def __init__(self, text_encoder_model_pretrained, audio_encoder_model_pretrained, tokenizer, processor):
        super().__init__()

        
        text_encoder  = ModalityEncoder(text_encoder_model_pretrained,  SelfAttentivePooling(768).half())
        audio_encoder = ModalityEncoder(audio_encoder_model_pretrained, SelfAttentivePooling(768).half())

        self.text_encoder  = text_encoder
        self.audio_encoder = audio_encoder
        self.tokenizer     = tokenizer
        self.processor     = processor

    def embed_text(self, texts):
        # texts is list[B][T] -- flatten to a single batch so the encoder sees B*T items
        B = len(texts)
        T = len(texts[0])
        flat_texts = [texts[b][t] for b in range(B) for t in range(T)]

        tokens = encode_text_inputs(flat_texts, self.tokenizer)
        with torch.no_grad():
            hidden = self.text_encoder.backbone(**tokens).last_hidden_state
        flat_emb = self.text_encoder(hidden, tokens["attention_mask"])  # (B*T, d)

        return flat_emb.view(B, T, -1)  # (B, T, d)

    def embed_audio(self, audio, lengths, text_only=None):
        # audio: (B, T, samples), lengths: (B, T)
        B, T, _ = audio.shape
        d_model  = 768

        device   = audio.device

        out = torch.zeros(B, T, d_model, dtype=audio.dtype, device=device)

        for t in range(T):
            # skip turns where every item in the batch has no audio
            if text_only is not None and text_only[:, t].all():
                continue

            audio_t   = audio[:, t, :]    # (B, samples)
            lengths_t = lengths[:, t]     # (B,)

            inputs, lengths_t = encode_audio_inputs(audio_t, lengths_t, self.processor)
            with torch.no_grad():
                hidden = self.audio_encoder.backbone(**inputs).last_hidden_state

            T_out    = hidden.shape[1]
            raw_mask = (torch.arange(audio_t.shape[1], device = device).unsqueeze(0)
                        < lengths_t.unsqueeze(1).to(device = device)).float()
            mask_ds  = F.interpolate(raw_mask.unsqueeze(1), size=T_out, mode="nearest").squeeze(1).bool()

            emb = self.audio_encoder(hidden, mask_ds)  # (B, d)
            out[:, t, :] = emb

        return out  # (B, T, d)


    def forward(self, audio, lengths, texts, text_only=None):
        text_emb  = self.embed_text(texts)
        audio_emb = self.embed_audio(audio, lengths, text_only)
        return text_emb, audio_emb  # both (B, T, d)
