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

import math
import random
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import act_to_func


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: Optional[float] = 0.1,
        activation_name: Optional[str] = "gelu",
    ) -> None:
        super(FFN, self).__init__()
        self.d_model = d_model
        self.activation_name = activation_name
        self.dim_feedforward = dim_feedforward
        self.layer_inc = nn.Linear(d_model, dim_feedforward)
        self.layer_reduce = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hid_state = self.layer_inc(hidden)
        hid_state = self.dropout(act_to_func(self.activation_name)(hid_state))
        return self.layer_reduce(hid_state)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: Optional[float] = 0.1,
        query_input_dim: Optional[int] = None,
        key_input_dim: Optional[int] = None,
        value_input_dim: Optional[int] = None,
        context_len: Optional[int] = None,
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.hidden_per_head = d_model // n_heads
        self.scale = self.hidden_per_head ** 0.5
        self.query_input_dim = d_model if query_input_dim is None else query_input_dim
        self.key_input_dim = d_model if key_input_dim is None else key_input_dim
        self.value_input_dim = d_model if value_input_dim is None else value_input_dim

        self.key_projection = nn.Linear(self.key_input_dim, d_model)
        self.query_projection = nn.Linear(self.query_input_dim, d_model)
        self.value_projection = nn.Linear(self.value_input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # attn_type in ["encoder", "decoder"]
        self.attn_type = "encoder"

        if context_len is not None:
            self.attn_type = "decoder"
            self.register_buffer("masked_val", torch.FloatTensor([-1e4]))
            self.register_buffer(
                "tri_mask",
                torch.tril(torch.ones((context_len, context_len), dtype=torch.uint8)).view(
                    1, 1, context_len, context_len
                ),
            )
        else:
            self.register_buffer("masked_val", None)
            self.register_buffer("tri_mask", None)

    def _transpose_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, emb_dim) ->
        # (batch_size, seq_len, n_heads, hidden_per_head)
        new_shape = x.shape[:-1] + (self.n_heads, self.hidden_per_head)
        x = x.view(*new_shape)
        # (batch_size, seq_len, n_heads, hidden_per_head) ->
        # -> (batch_size, n_heads, seq_len, hidden_per_head)
        return x.permute(0, 2, 1, 3)

    def _generate_decoder_self_attn_mask(self, q_seq_len: int, k_seq_len: int) -> torch.Tensor:
        # TODO: fix case then k_seq_len < q_seq_len
        if self.training:
            attn_mask = self.tri_mask[:, :, k_seq_len - q_seq_len : k_seq_len, :k_seq_len]
        else:
            attn_mask = torch.ones((1, 1, k_seq_len, k_seq_len)).type_as(self.tri_mask)
        return attn_mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        get_attention_scores: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        query_proj = self.query_projection(query)
        key_proj = self.key_projection(key)
        value_proj = self.value_projection(value)

        query_proj = self._transpose_to_heads(query_proj)
        key_proj = self._transpose_to_heads(key_proj)
        value_proj = self._transpose_to_heads(value_proj)

        raw_scores = torch.matmul(query_proj, key_proj.transpose(-1, -2))

        if self.attn_type == "decoder":
            decoder_attn_mask = self._generate_decoder_self_attn_mask(key_proj.shape[2], key_proj.shape[2])
            raw_scores = torch.where(
                decoder_attn_mask.bool(),
                raw_scores,
                self.masked_val.to(raw_scores.dtype),
            )

        if attention_mask is not None:
            raw_scores = raw_scores + attention_mask
        attn_scores = F.softmax(raw_scores / self.scale, dim=-1)
        attn = torch.matmul(attn_scores, value_proj)
        attn = attn.permute(0, 2, 1, 3).contiguous()
        new_shape = attn.shape[:-2] + (self.d_model,)
        attn = attn.view(*new_shape)
        output = (attn,)
        if get_attention_scores:
            output = output + (attn_scores,)
        return output


class SinusoidalPositionalEmbedding(nn.Module):
    # based on implenemtation from fairseq
    def __init__(self, context_len: int, embedding_dim: int):
        super(SinusoidalPositionalEmbedding, self).__init__()
        # im not shure about +1 operation
        self.context_len = context_len + 1
        self.embedding_dim = embedding_dim

        half_dim = embedding_dim // 2
        weight = math.log(10000) / (half_dim - 1)
        weight = torch.exp(torch.arange(half_dim, dtype=torch.float) * -weight)
        weight = torch.arange(self.context_len, dtype=torch.float).unsqueeze(1) * weight.unsqueeze(0)
        weight = torch.cat([torch.sin(weight), torch.cos(weight)], dim=1).view(self.context_len, -1)
        if embedding_dim % 2 == 1:
            weight = torch.cat([weight, torch.zeros(self.context_len, 1)], dim=1)
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(
        self,
        input_positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = input_positions.shape
        # im not shure about +1 operation
        # TODO: find more information about it
        input_positions = input_positions + 1
        inds = self.weight.index_select(0, input_positions.view(-1)).view(batch_size, seq_len, -1).detach()
        return inds


class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_length: int,
        pad_token_id: int,
        token_type_vocab_size: int,
        pos_embedding_type: Optional[str] = "embedding",
        dropout: Optional[float] = 0.1,
        use_layer_norm: Optional[bool] = False,
        ln_eps: Optional[float] = 1e-12,
    ) -> None:
        super(TransformerEmbedding, self).__init__()
        assert token_type_vocab_size >= 0
        assert vocab_size >= 0
        assert d_model >= 0
        assert max_length > 0
        if token_type_vocab_size == 0:
            use_token_type_embeddings = False
        else:
            use_token_type_embeddings = True
        assert pos_embedding_type in ["embedding", "timing"]

        self.token_embedding = nn.Embedding(vocab_size, d_model, pad_token_id)

        if pos_embedding_type == "embedding":
            self.positional_embedding = nn.Embedding(max_length, d_model)
        else:
            self.positional_embedding = SinusoidalPositionalEmbedding(max_length, d_model)
        if use_token_type_embeddings:
            self.token_type_embedding = nn.Embedding(token_type_vocab_size, d_model)
        else:
            self.token_type_embedding = nn.Identity()
            self.token_type_embedding.register_parameter("weight", None)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_ids", torch.arange(max_length).expand((1, -1)))

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(d_model, eps=ln_eps)
        else:
            self.layer_norm = nn.Identity()
            self.layer_norm.register_parameter("weight", None)
            self.layer_norm.register_parameter("bias", None)

        self.use_layer_norm = use_layer_norm
        self.use_token_type_embeddings = use_token_type_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        token_emb = self.token_embedding(input_ids)
        input_shape = token_emb.shape
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape[:-1],
                dtype=torch.long,
                device=self.pos_ids.device,
            )
        tt_emb = self.token_type_embedding(token_type_ids)
        if self.use_token_type_embeddings:
            token_emb = token_emb + tt_emb

        pos_ids = self.pos_ids[:, : input_shape[1]]

        token_emb = token_emb + self.positional_embedding(pos_ids)
        token_emb = self.layer_norm(token_emb)
        token_emb = self.dropout(token_emb)
        return token_emb


class BaseTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        encoder_class: Callable,
        layerdrop_threshold: Optional[float] = 0.0,
        **kwargs,
    ) -> None:
        super(BaseTransformerEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList([encoder_class(**kwargs) for _ in range(num_layers)])
        self.layerdrop_threshold = layerdrop_threshold

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        get_attention_scores: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        attn_scores = []
        for layer in self.encoder_layers:
            # LayerDrop: https://arxiv.org/abs/1909.11556
            # TODO: make something with different shapes
            # of attn_scores while using LayerDrop
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop_threshold):
                continue
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                get_attention_scores=get_attention_scores,
            )
            if get_attention_scores:
                attn_scores.append(hidden[1])
            hidden = hidden[0]
        output = (hidden,)
        if get_attention_scores:
            attn_scores = torch.stack(attn_scores, dim=-1)
            output = output + (attn_scores,)
        return output


class BaseTransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        decoder_class: Callable,
        layerdrop_threshold: Optional[float] = 0.0,
        **kwargs,
    ) -> None:
        super(BaseTransformerDecoder, self).__init__()
        self.decoder_layers = nn.ModuleList([decoder_class(**kwargs) for _ in range(num_layers)])
        self.layerdrop_threshold = layerdrop_threshold

    def forward(
        self,
        hidden: torch.Tensor,
        encoder_hidden_state: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        get_attention_scores: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        attn_scores = []
        for layer in self.decoder_layers:
            # LayerDrop: https://arxiv.org/abs/1909.11556
            # TODO: make something with different shapes
            # of attn_scores while using LayerDrop
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop_threshold):
                continue
            hidden = layer(
                hidden=hidden,
                encoder_hidden_state=encoder_hidden_state,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                get_attention_scores=get_attention_scores,
            )
            if get_attention_scores:
                attn_scores.append(hidden[1])
            hidden = hidden[0]

        output = (hidden,)
        if get_attention_scores:
            attn_scores = torch.stack(attn_scores, dim=-1)
            output = output + (attn_scores,)
        return output
