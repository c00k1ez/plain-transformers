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

from typing import Optional, Callable, Dict, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: add more flexible opportunity to init encoder & decoder
# TODO: add label smoothing loss
# TODO: write more complex solution for embedding sharing
class MultimodalTransformer(nn.Module):
    def __init__(
        self,
        first_encoder_class: Callable,
        second_encoder_class: Callable,
        decoder_class: Callable,
        first_encoder_vocab_size: int,
        second_encoder_vocab_size: int,
        decoder_vocab_size: int,
        use_token_type_embeddings: Optional[bool] = False,
        share_decoder_head_weights: Optional[bool] = True,
        share_encoder_decoder_embeddings: Optional[bool] = False,
        share_encoder_embeddings: Optional[bool] = False,
        **kwargs
    ) -> None:
        super(MultimodalTransformer, self).__init__()
        self.first_encoder = first_encoder_class(
            **kwargs,
            use_token_type_embeddings=use_token_type_embeddings,
            vocab_size=first_encoder_vocab_size
        )
        self.second_encoder = second_encoder_class(
            **kwargs,
            use_token_type_embeddings=use_token_type_embeddings,
            vocab_size=second_encoder_vocab_size
        )

        self.decoder = decoder_class(**kwargs, vocab_size=decoder_vocab_size)
        self.lm_head = nn.Linear(
            kwargs["d_model"], decoder_vocab_size, bias=False
        )
        if share_decoder_head_weights:
            self.lm_head.weight = self.decoder.embedding.token_embedding.weight
        if share_encoder_decoder_embeddings:
            self.first_encoder.embedding.token_embedding.weight = (
                self.decoder.embedding.token_embedding.weight
            )
            self.second_encoder.embedding.token_embedding.weight = (
                self.decoder.embedding.token_embedding.weight
            )
        if share_encoder_embeddings:
            self.first_encoder.embedding.token_embedding.weight = (
                self.second_encoder.embedding.token_embedding.weight
            )
        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=kwargs["pad_token_id"]
        )
        self.pad_token_id = kwargs["pad_token_id"]

    def forward(
        self,
        first_encoder_input_ids: torch.Tensor,
        second_encoder_input_ids: torch.Tensor,
        labels: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        first_encoder_attention_mask: Optional[torch.Tensor] = None,
        second_encoder_attention_mask: Optional[torch.Tensor] = None,
        get_attention_scores: Optional[bool] = False,
        cached_encoder_state: Optional[
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ] = None,
        return_encoder_state: Optional[bool] = False,
        compute_loss: Optional[bool] = False,
    ) -> Dict[
        str,
        Union[
            torch.Tensor,
            Dict[str, torch.Tensor],
            Dict[str, Dict[str, torch.Tensor]],
        ],
    ]:
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

        output = {"lm_probs": torch.softmax(raw_probs, dim=-1)}
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
                    torch.LongTensor(
                        [[self.pad_token_id]], device=labels.device
                    ).repeat(batch_size, 1),
                ],
                dim=-1,
            )
            output["loss_val"] = self.loss_function(
                raw_probs.permute(0, 2, 1), labels
            )
        return output
