# Copyright 2021 c00k1ez (https://github.com/c00k1ez). All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


class PreLNEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: Optional[float] = 0.1,
        activation_name: Optional[str] = "gelu",
        ln_eps: Optional[float] = 1e-12,
    ) -> None:
        super(PreLNEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FFN(d_model, dim_feedforward, dropout, activation_name=activation_name)
        self.pre_attn_ln = nn.LayerNorm(d_model, eps=ln_eps)
        self.pre_ffn_ln = nn.LayerNorm(d_model, eps=ln_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        get_attention_scores: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:

        attn_scores = None
        block_state = self.pre_attn_ln(hidden)
        block_state = self.self_attention(
            query=block_state,
            key=block_state,
            value=block_state,
            attention_mask=attention_mask,
            get_attention_scores=get_attention_scores,
        )
        if get_attention_scores:
            attn_scores = block_state[1]
        block_state = block_state[0]
        block_state = block_state + hidden

        ffn_block_state = self.pre_ffn_ln(block_state)
        ffn_block_state = self.ffn(ffn_block_state)
        ffn_block_state = ffn_block_state + block_state

        output = (ffn_block_state,)
        if get_attention_scores:
            output = output + (attn_scores,)
        return output


# TODO: think about merge matrix for attention
class PreLNTransformerEncoder(nn.Module):
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
        layerdrop_threshold: Optional[float] = 0.0,
    ) -> None:
        super(PreLNTransformerEncoder, self).__init__()
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
        )
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            encoder_class=PreLNEncoderLayer,
            d_model=d_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation_name=activation_name,
            ln_eps=ln_eps,
            layerdrop_threshold=layerdrop_threshold,
        )

        self.post_encoder_ln = nn.LayerNorm(d_model, eps=ln_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        get_attention_scores: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:

        attention_mask = create_attention_mask(attention_mask, input_ids.shape, input_ids.device)
        embeddings = self.embedding(input_ids, token_type_ids)
        hidden = self.encoder(
            embeddings,
            attention_mask=attention_mask,
            get_attention_scores=get_attention_scores,
        )
        hidden_ln = self.post_encoder_ln(hidden[0])
        output = (hidden_ln,)
        if get_attention_scores:
            output = output + (hidden[1],)
        return output
