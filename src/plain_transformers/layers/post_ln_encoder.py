from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_layers import (
    FFN,
    MultiHeadAttention,
    TransformerEmbedding,
    TransformerEncoder,
)
from .utils import create_attention_mask


class PostLNEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: Optional[float] = 0.1,
        activation_name: Optional[str] = "gelu",
        ln_eps: Optional[float] = 1e-12,
    ) -> None:
        super(PostLNEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FFN(
            d_model, dim_feedforward, dropout, activation_name=activation_name
        )
        self.post_attn_ln = nn.LayerNorm(d_model, eps=ln_eps)
        self.post_ffn_ln = nn.LayerNorm(d_model, eps=ln_eps)

        self.merge_matrix = nn.Linear(d_model, d_model)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        get_attention_scores: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:

        attn_scores = None
        block_state = self.self_attention(
            query=hidden,
            key=hidden,
            value=hidden,
            attention_mask=attention_mask,
            get_attention_scores=get_attention_scores,
        )
        if get_attention_scores:
            attn_scores = block_state[1]
        block_state = block_state[0]
        block_state = self.merge_matrix(block_state)
        block_state = block_state + hidden
        block_state = self.post_attn_ln(block_state)

        ffn_block_state = self.ffn(block_state)
        ffn_block_state = ffn_block_state + block_state
        ffn_block_state = self.post_ffn_ln(ffn_block_state)

        output = (ffn_block_state,)
        if get_attention_scores:
            output = output + (attn_scores,)
        return output


class PostLNTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        max_length: int,
        pad_token_id: int,
        token_type_vocab_size: int,
        n_heads: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: Optional[float] = 0.1,
        use_embedding_layer_norm: Optional[bool] = False,
        pos_embedding_type: Optional[str] = "embedding",
        activation_name: Optional[str] = "gelu",
        ln_eps: Optional[float] = 1e-12,
        use_token_type_embeddings: Optional[bool] = True,
    ) -> None:
        super(PostLNTransformerEncoder, self).__init__()
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_length=max_length,
            pad_token_id=pad_token_id,
            token_type_vocab_size=token_type_vocab_size,
            pos_embedding_type=pos_embedding_type,
            dropout=dropout,
            use_layer_norm=use_embedding_layer_norm,
            ln_eps=ln_eps,
            use_token_type_embeddings=use_token_type_embeddings,
        )
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            encoder_class=PostLNEncoderLayer,
            d_model=d_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation_name=activation_name,
            ln_eps=ln_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        get_attention_scores: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:

        attention_mask = create_attention_mask(
            attention_mask, input_ids.shape, input_ids.device
        )
        embeddings = self.embedding(input_ids, token_type_ids)
        hidden = self.encoder(
            embeddings,
            attention_mask=attention_mask,
            get_attention_scores=get_attention_scores,
        )
        return hidden
