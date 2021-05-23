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

from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from plain_transformers.losses import LabelSmoothingLoss


# TODO: write more complex solution for embedding sharing
class MultimodalTransformer(nn.Module):
    def __init__(
        self,
        first_encoder_class: Callable,
        second_encoder_class: Callable,
        decoder_class: Callable,
        d_model: int,
        first_encoder_vocab_size: int,
        first_encoder_max_length: int,
        first_encoder_pad_token_id: int,
        first_encoder_token_type_vocab_size: int,
        first_encoder_n_heads: int,
        first_encoder_dim_feedforward: int,
        first_encoder_num_layers: int,
        second_encoder_vocab_size: int,
        second_encoder_max_length: int,
        second_encoder_pad_token_id: int,
        second_encoder_token_type_vocab_size: int,
        second_encoder_n_heads: int,
        second_encoder_dim_feedforward: int,
        second_encoder_num_layers: int,
        decoder_max_length: int,
        decoder_vocab_size: int,
        decoder_pad_token_id: int,
        decoder_token_type_vocab_size: int,
        decoder_n_heads: int,
        decoder_dim_feedforward: int,
        decoder_num_layers: int,
        decoder_use_embedding_layer_norm: Optional[bool] = True,
        decoder_pos_embedding_type: Optional[str] = "embedding",
        decoder_activation_name: Optional[str] = "gelu",
        decoder_ln_eps: Optional[float] = 1e-12,
        decoder_dropout: Optional[float] = 0.1,
        decoder_layerdrop_threshold: Optional[float] = 0.0,
        decoder_type: Optional[str] = "post_ln",
        first_encoder_dropout: Optional[float] = 0.1,
        first_encoder_use_embedding_layer_norm: Optional[bool] = True,
        first_encoder_pos_embedding_type: Optional[str] = "embedding",
        first_encoder_activation_name: Optional[str] = "gelu",
        first_encoder_ln_eps: Optional[float] = 1e-12,
        first_encoder_layerdrop_threshold: Optional[float] = 0.0,
        first_encoder_type: Optional[str] = "post_ln",
        second_encoder_dropout: Optional[float] = 0.1,
        second_encoder_use_embedding_layer_norm: Optional[bool] = True,
        second_encoder_pos_embedding_type: Optional[str] = "embedding",
        second_encoder_activation_name: Optional[str] = "gelu",
        second_encoder_ln_eps: Optional[float] = 1e-12,
        second_encoder_layerdrop_threshold: Optional[float] = 0.0,
        second_encoder_type: Optional[str] = "post_ln",
        share_decoder_head_weights: Optional[bool] = True,
        share_encoder_decoder_embeddings: Optional[bool] = False,
        share_encoder_embeddings: Optional[bool] = False,
        label_smoothing: Optional[float] = 0.0,
    ) -> None:
        super(MultimodalTransformer, self).__init__()
        self.first_encoder = first_encoder_class(
            d_model=d_model,
            max_length=first_encoder_max_length,
            pad_token_id=first_encoder_pad_token_id,
            token_type_vocab_size=first_encoder_token_type_vocab_size,
            n_heads=first_encoder_n_heads,
            dim_feedforward=first_encoder_dim_feedforward,
            num_layers=first_encoder_num_layers,
            dropout=first_encoder_dropout,
            use_embedding_layer_norm=first_encoder_use_embedding_layer_norm,
            pos_embedding_type=first_encoder_pos_embedding_type,
            activation_name=first_encoder_activation_name,
            ln_eps=first_encoder_ln_eps,
            vocab_size=first_encoder_vocab_size,
            layerdrop_threshold=first_encoder_layerdrop_threshold,
            encoder_type=first_encoder_type,
        )
        self.second_encoder = second_encoder_class(
            d_model=d_model,
            max_length=second_encoder_max_length,
            pad_token_id=second_encoder_pad_token_id,
            token_type_vocab_size=second_encoder_token_type_vocab_size,
            n_heads=second_encoder_n_heads,
            dim_feedforward=second_encoder_dim_feedforward,
            num_layers=second_encoder_num_layers,
            dropout=second_encoder_dropout,
            use_embedding_layer_norm=second_encoder_use_embedding_layer_norm,
            pos_embedding_type=second_encoder_pos_embedding_type,
            activation_name=second_encoder_activation_name,
            ln_eps=second_encoder_ln_eps,
            vocab_size=second_encoder_vocab_size,
            layerdrop_threshold=second_encoder_layerdrop_threshold,
            encoder_type=second_encoder_type,
        )

        self.decoder = decoder_class(
            d_model=d_model,
            vocab_size=decoder_vocab_size,
            max_length=decoder_max_length,
            pad_token_id=decoder_pad_token_id,
            token_type_vocab_size=decoder_token_type_vocab_size,
            n_heads=decoder_n_heads,
            dim_feedforward=decoder_dim_feedforward,
            num_layers=decoder_num_layers,
            dropout=decoder_dropout,
            use_embedding_layer_norm=decoder_use_embedding_layer_norm,
            pos_embedding_type=decoder_pos_embedding_type,
            activation_name=decoder_activation_name,
            ln_eps=decoder_ln_eps,
            layerdrop_threshold=decoder_layerdrop_threshold,
            decoder_type=decoder_type,
        )
        self.lm_head = nn.Linear(d_model, decoder_vocab_size, bias=False)
        if share_decoder_head_weights:
            self.lm_head.weight = self.decoder.embedding.token_embedding.weight
        if share_encoder_decoder_embeddings:
            self.first_encoder.embedding.token_embedding.weight = self.decoder.embedding.token_embedding.weight
            self.second_encoder.embedding.token_embedding.weight = self.decoder.embedding.token_embedding.weight
        if share_encoder_embeddings:
            self.first_encoder.embedding.token_embedding.weight = self.second_encoder.embedding.token_embedding.weight
        self.loss_function = LabelSmoothingLoss(
            smoothing=label_smoothing,
            ignore_index=decoder_pad_token_id,
        )
        self.pad_token_id = decoder_pad_token_id

    def forward(
        self,
        first_encoder_input_ids: torch.Tensor,
        second_encoder_input_ids: torch.Tensor,
        labels: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        first_encoder_attention_mask: Optional[torch.Tensor] = None,
        second_encoder_attention_mask: Optional[torch.Tensor] = None,
        get_attention_scores: Optional[bool] = False,
        cached_encoder_state: Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = None,
        return_encoder_state: Optional[bool] = False,
        compute_loss: Optional[bool] = False,
        get_logits: Optional[bool] = False,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]],],]:
        first_encoder_state, second_encoder_state = None, None
        attn_scores = {}
        if cached_encoder_state is not None:
            first_encoder_state, second_encoder_state = cached_encoder_state
        else:
            first_encoder_state = self.first_encoder(
                input_ids=first_encoder_input_ids,
                attention_mask=first_encoder_attention_mask,
                token_type_ids=None,
                get_attention_scores=get_attention_scores,
            )

            second_encoder_state = self.second_encoder(
                input_ids=second_encoder_input_ids,
                attention_mask=second_encoder_attention_mask,
                token_type_ids=None,
                get_attention_scores=get_attention_scores,
            )
            if get_attention_scores:
                attn_scores["encoder"] = {
                    "first_encoder": first_encoder_state[1],
                    "second_encoder": second_encoder_state[1],
                }

            first_encoder_state = {
                "key": first_encoder_state[0],
                "value": first_encoder_state[0],
            }
            second_encoder_state = {
                "key": second_encoder_state[0],
                "value": second_encoder_state[0],
            }
        hidden = self.decoder(
            input_ids=labels,
            first_encoder_hidden_state=first_encoder_state,
            second_encoder_hidden_state=second_encoder_state,
            attention_mask=decoder_attention_mask,
            first_encoder_attention_mask=first_encoder_attention_mask,
            second_encoder_attention_mask=second_encoder_attention_mask,
            get_attention_scores=get_attention_scores,
        )
        if get_attention_scores:
            attn_scores["decoder"] = hidden[1]
        hidden = hidden[0]
        raw_probs = self.lm_head(hidden)

        output = {"lm_probs": torch.softmax(raw_probs, dim=-1) if not get_logits else raw_probs}
        if return_encoder_state:
            output["encoder_hidden_state"] = (
                first_encoder_state,
                second_encoder_state,
            )
        if compute_loss:
            batch_size = labels.shape[0]
            labels = torch.cat(
                [
                    labels[:, 1:],
                    torch.LongTensor([[self.pad_token_id]]).to(labels.device).repeat(batch_size, 1),
                ],
                dim=-1,
            )
            output["loss_val"] = self.loss_function(raw_probs.view(-1, raw_probs.shape[-1]), labels.view(-1))
        return output
