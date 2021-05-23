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

from .common_layers import BaseTransformerEncoder, TransformerEmbedding
from .post_ln_encoder import PostLNEncoderLayer
from .pre_ln_encoder import PreLNEncoderLayer
from .utils import create_attention_mask


class TransformerEncoder(nn.Module):
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
        use_attention_merge_matrix: Optional[bool] = True,
        encoder_type: Optional[str] = "post_ln",
    ) -> None:
        super(TransformerEncoder, self).__init__()
        assert encoder_type in ["post_ln", "pre_ln"]
        self.encoder_type = encoder_type
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
        encoder_class = PostLNEncoderLayer
        if encoder_type == "pre_ln":
            encoder_class = PreLNEncoderLayer
        self.encoder = BaseTransformerEncoder(
            num_layers=num_layers,
            encoder_class=encoder_class,
            d_model=d_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation_name=activation_name,
            ln_eps=ln_eps,
            layerdrop_threshold=layerdrop_threshold,
            use_attention_merge_matrix=use_attention_merge_matrix,
        )
        if encoder_type == "pre_ln":
            self.post_encoder_ln = nn.LayerNorm(d_model, eps=ln_eps)
        else:
            self.post_encoder_ln = nn.Identity()
            self.post_encoder_ln.register_parameter("weight", None)
            self.post_encoder_ln.register_parameter("bias", None)

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
