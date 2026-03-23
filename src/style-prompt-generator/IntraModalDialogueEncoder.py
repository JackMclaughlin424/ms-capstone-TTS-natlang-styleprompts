import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional


class IntraSpeakerTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_encoder_layers: int
                 , dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    def forward(self, turn_embeddings: torch.Tensor, speaker_ids_list: List[List[str]]):
        B, seq_len, _ = turn_embeddings.shape
        device = turn_embeddings.device

        # Generate causal mask and ensure it has the same dtype as turn_embeddings
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device).to(turn_embeddings.dtype)

        # Create numerical speaker IDs for the current batch
        numerical_speaker_ids = []
        global_speaker_id_map = {}
        next_id = 0

        for b_idx in range(B):
            chain_speakers = speaker_ids_list[b_idx]
            chain_numerical_ids = []
            for speaker_str in chain_speakers:
                if speaker_str not in global_speaker_id_map:
                    global_speaker_id_map[speaker_str] = next_id
                    next_id += 1
                chain_numerical_ids.append(global_speaker_id_map[speaker_str])
            numerical_speaker_ids.append(chain_numerical_ids)

        # Convert to tensor (B, seq_len)
        speaker_ids_tensor = torch.tensor(numerical_speaker_ids, dtype=torch.long, device=device)

        # Create same speaker boolean mask (B, seq_len, seq_len)
        same_speaker_bool_mask = (speaker_ids_tensor.unsqueeze(2) == speaker_ids_tensor.unsqueeze(1))

        # Intra-speaker mask: allow attention within the same speaker, block attention to different speakers
        # Additive mask: 0 for allowed, -inf for blocked
        inf_val = torch.finfo(turn_embeddings.dtype).min
        # Explicitly cast 0. to the correct dtype for torch.where
        zero_tensor = torch.tensor(0., dtype=turn_embeddings.dtype, device=device)
        intra_speaker_additive_mask = torch.where(same_speaker_bool_mask, zero_tensor, inf_val)

        # Combine causal mask with intra-speaker mask
        # Causal mask is (S, S), intra_speaker_additive_mask is (B, S, S)
        # Unsqueezing causal_mask to (1, S, S) allows broadcasting
        final_mask = causal_mask.unsqueeze(0) + intra_speaker_additive_mask

        # Reshape the mask to (B * nhead, S, S)
        nhead_local = self.transformer_encoder.layers[0].self_attn.num_heads
        final_mask = final_mask.unsqueeze(1).expand(-1, nhead_local, -1, -1).reshape(B * nhead_local, seq_len, seq_len)

        output = self.transformer_encoder(turn_embeddings, mask=final_mask)
        return output



class InterSpeakerTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_encoder_layers: int
                 , dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    def forward(self, turn_embeddings: torch.Tensor, speaker_ids_list: List[List[str]]):
        B, seq_len, _ = turn_embeddings.shape
        device = turn_embeddings.device

        # Generate causal mask and ensure it has the same dtype as turn_embeddings
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device).to(turn_embeddings.dtype)

        # Create numerical speaker IDs for the current batch
        numerical_speaker_ids = []
        global_speaker_id_map = {}
        next_id = 0

        for b_idx in range(B):
            chain_speakers = speaker_ids_list[b_idx]
            chain_numerical_ids = []
            for speaker_str in chain_speakers:
                if speaker_str not in global_speaker_id_map:
                    global_speaker_id_map[speaker_str] = next_id
                    next_id += 1
                chain_numerical_ids.append(global_speaker_id_map[speaker_str])
            numerical_speaker_ids.append(chain_numerical_ids)

        # Convert to tensor (B, seq_len)
        speaker_ids_tensor = torch.tensor(numerical_speaker_ids, dtype=torch.long, device=device)

        # Create same speaker boolean mask (B, seq_len, seq_len)
        same_speaker_bool_mask = (speaker_ids_tensor.unsqueeze(2) == speaker_ids_tensor.unsqueeze(1))

        # Inter-speaker mask: allow attention between different speakers, block attention to the same speaker
        # Additive mask: 0 for allowed, -inf for blocked
        inf_val = torch.finfo(turn_embeddings.dtype).min
        # Explicitly cast 0. to the correct dtype for torch.where
        zero_tensor = torch.tensor(0., dtype=turn_embeddings.dtype, device=device)
        inter_speaker_additive_mask = torch.where(~same_speaker_bool_mask, zero_tensor, inf_val)

        # Combine causal mask with inter-speaker mask
        final_mask = causal_mask.unsqueeze(0) + inter_speaker_additive_mask

        # Reshape the mask to (B * nhead, S, S)
        nhead_local = self.transformer_encoder.layers[0].self_attn.num_heads
        final_mask = final_mask.unsqueeze(1).expand(-1, nhead_local, -1, -1).reshape(B * nhead_local, seq_len, seq_len)

        output = self.transformer_encoder(turn_embeddings, mask=final_mask)
        return output
    
class SpeakerAwareTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_encoder_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        # Instantiate separate ContextAwareTransformer for intra-speaker masking
        self.intra_transformer = IntraSpeakerTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        # Instantiate separate ContextAwareTransformer for inter-speaker masking
        self.inter_transformer = InterSpeakerTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

    def forward(self, turn_embeddings: torch.Tensor, speaker_ids_list: List[List[str]]):
        # Pass through the intra-speaker context aware transformer
        intra_emb = self.intra_transformer(
            turn_embeddings,
            speaker_ids_list=speaker_ids_list
        )

        # Pass through the inter-speaker context aware transformer
        inter_emb = self.inter_transformer(
            turn_embeddings,
            speaker_ids_list=speaker_ids_list
        )
        return intra_emb, inter_emb
    

class ContextAwareTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_encoder_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        # TransformerEncoderLayer is the basic building block for the encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Set batch_first to True for (B, S, E) input
        )
        # TransformerEncoder stacks multiple EncoderLayers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    def forward(self, turn_embeddings: torch.Tensor):
        # Input `turn_embeddings` is expected to be (Batch_Size, Num_Turns, Embedding_Dim) which is (B, S, E)

        # Generate causal mask: ensures each position only attends to previous positions
        seq_len = turn_embeddings.shape[1] # Sequence length is the second dimension for batch_first=True
        src_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(turn_embeddings.device)

        # Pass through the Transformer Encoder with the mask
        output = self.transformer_encoder(turn_embeddings, mask=src_mask)

        # Output is already (Batch_Size, Num_Turns, Embedding_Dim) because batch_first=True
        return output
