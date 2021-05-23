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

from .common_layers import BaseTransformerDecoder, TransformerEmbedding
from .post_ln_decoder import PostLNDecoderLayer, PostLNMultimodalDecoderLayer
from .utils import create_attention_mask


# TODO: implement pre ln decoder
class TransformerDecoder(nn.Module):
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
        decoder_type: Optional[str] = "post_ln",
    ) -> None:
        super(TransformerDecoder, self).__init__()
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_length=max_length,
            pad_token_id=pad_token_id,
            token_type_vocab_size=token_type_vocab_size,
            dropout=dropout,
            use_layer_norm=use_embedding_layer_norm,
            ln_eps=ln_eps,
            pos_embedding_type=pos_embedding_type,
        )

        self.decoder = BaseTransformerDecoder(
            num_layers=num_layers,
            decoder_class=PostLNDecoderLayer,
            d_model=d_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation_name=activation_name,
            ln_eps=ln_eps,
            context_len=max_length,
            layerdrop_threshold=layerdrop_threshold,
            use_attention_merge_matrix=use_attention_merge_matrix,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        get_attention_scores: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:

        attention_mask = create_attention_mask(
            attention_mask=attention_mask,
            input_shape=input_ids.shape,
            device=input_ids.device,
        )

        encoder_attention_mask = create_attention_mask(
            attention_mask=encoder_attention_mask,
            input_shape=encoder_hidden_state["key"].shape[:-1],
            device=encoder_hidden_state["key"].device,
            src_size=input_ids.shape[-1],
        )

        embeddings = self.embedding(input_ids)

        hidden = self.decoder(
            hidden=embeddings,
            encoder_hidden_state=encoder_hidden_state,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            get_attention_scores=get_attention_scores,
        )

        return hidden


class MultimodalTransformerDecoder(nn.Module):
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
        decoder_type: Optional[str] = "post_ln",
    ) -> None:
        super(MultimodalTransformerDecoder, self).__init__()
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_length=max_length,
            pad_token_id=pad_token_id,
            token_type_vocab_size=token_type_vocab_size,
            dropout=dropout,
            use_layer_norm=use_embedding_layer_norm,
            ln_eps=ln_eps,
            pos_embedding_type=pos_embedding_type,
        )

        self.decoder = BaseTransformerDecoder(
            num_layers=num_layers,
            decoder_class=PostLNMultimodalDecoderLayer,
            d_model=d_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation_name=activation_name,
            ln_eps=ln_eps,
            context_len=max_length,
            layerdrop_threshold=layerdrop_threshold,
            use_attention_merge_matrix=use_attention_merge_matrix,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        first_encoder_hidden_state: torch.Tensor,
        second_encoder_hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        first_encoder_attention_mask: Optional[torch.Tensor] = None,
        second_encoder_attention_mask: Optional[torch.Tensor] = None,
        get_attention_scores: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        attention_mask = create_attention_mask(
            attention_mask=attention_mask,
            input_shape=input_ids.shape,
            device=input_ids.device,
        )

        first_encoder_attention_mask = create_attention_mask(
            attention_mask=first_encoder_attention_mask,
            input_shape=first_encoder_hidden_state["key"].shape[:-1],
            device=first_encoder_hidden_state["key"].device,
            src_size=input_ids.shape[-1],
        )

        second_encoder_attention_mask = create_attention_mask(
            attention_mask=second_encoder_attention_mask,
            input_shape=second_encoder_hidden_state["key"].shape[:-1],
            device=second_encoder_hidden_state["key"].device,
            src_size=input_ids.shape[-1],
        )

        embeddings = self.embedding(input_ids)

        hidden = self.decoder(
            hidden=embeddings,
            encoder_hidden_state=(
                first_encoder_hidden_state,
                second_encoder_hidden_state,
            ),
            attention_mask=attention_mask,
            encoder_attention_mask=(
                first_encoder_attention_mask,
                second_encoder_attention_mask,
            ),
            get_attention_scores=get_attention_scores,
        )

        return hidden
