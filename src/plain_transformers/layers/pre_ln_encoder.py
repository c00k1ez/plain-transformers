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

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_layers import FFN, MultiHeadAttention


class PreLNEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: Optional[float] = 0.1,
        activation_name: Optional[str] = "gelu",
        ln_eps: Optional[float] = 1e-12,
        use_attention_merge_matrix: Optional[bool] = True,
    ) -> None:
        super(PreLNEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FFN(d_model, dim_feedforward, dropout, activation_name=activation_name)
        self.pre_attn_ln = nn.LayerNorm(d_model, eps=ln_eps)
        self.pre_ffn_ln = nn.LayerNorm(d_model, eps=ln_eps)
        if use_attention_merge_matrix:
            self.merge_matrix = nn.Linear(d_model, d_model)
        else:
            self.merge_matrix = nn.Identity()
            self.merge_matrix.register_parameter("weight", None)
            self.merge_matrix.register_parameter("bias", None)

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
        block_state = self.merge_matrix(block_state)
        block_state = block_state + hidden

        ffn_block_state = self.pre_ffn_ln(block_state)
        ffn_block_state = self.ffn(ffn_block_state)
        ffn_block_state = ffn_block_state + block_state

        output = (ffn_block_state,)
        if get_attention_scores:
            output = output + (attn_scores,)
        return output
