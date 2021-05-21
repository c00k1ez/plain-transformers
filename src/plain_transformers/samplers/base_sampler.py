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
import youtokentome as yttm
from plain_transformers import BPEWrapper


class BaseSampler(object):
    def __init__(
        self,
        model: nn.Module,
        encoder_tokenizer: Union[BPEWrapper, Tuple[BPEWrapper, BPEWrapper]],
        decoder_tokenizer: BPEWrapper,
        share_encoder_tokenizers: Optional[bool] = False,
        share_encoder_decoder_tokenizers: Optional[bool] = False,
    ) -> None:
        self.model = model
        self.first_encoder_tokenizer = None
        self.second_encoder_tokenizer = None
        self.decoder_tokenizer = decoder_tokenizer
        if isinstance(encoder_tokenizer, tuple):
            (
                self.first_encoder_tokenizer,
                self.second_encoder_tokenizer,
            ) = encoder_tokenizer
        else:
            self.first_encoder_tokenizer = encoder_tokenizer
        if share_encoder_tokenizers:
            self.first_encoder_tokenizer = self.second_encoder_tokenizer
        if share_encoder_decoder_tokenizers:
            self.decoder_tokenizer = self.first_encoder_tokenizer

    @torch.no_grad()
    def sample(self, logits: torch.Tensor, temperatype: float, **kwargs) -> torch.Tensor:
        raise NotImplementedError("You have to implement this method")

    @torch.no_grad()
    def generate(
        self,
        input_text,
        temperatype=1.0,
        max_length=50,
        second_input_text=None,
        decoder_input_text=None,
        device=None,
        **kwargs,
    ):
        if device is None:
            device = torch.device("cpu")
        if isinstance(input_text, torch.Tensor):
            assert len(input_text.shape) == 2
            encoder_input_ids = input_text.to(device)
        else:
            encoder_input_ids = torch.LongTensor(
                self.first_encoder_tokenizer.encode(
                    [input_text],
                    bos=True,
                    eos=True,
                    output_type=yttm.OutputType.ID,
                )
            ).to(device)
        second_encoder_input_ids = None
        if second_input_text is not None:
            if isinstance(second_input_text, torch.Tensor):
                assert len(second_input_text.shape) == 2
                second_encoder_input_ids = second_input_text.to(device)
            else:
                second_encoder_input_ids = torch.LongTensor(
                    self.second_encoder_tokenizer.encode(
                        [second_input_text],
                        bos=True,
                        eos=True,
                        output_type=yttm.OutputType.ID,
                    )
                ).to(device)

        labels = None
        if decoder_input_text is not None:
            if isinstance(decoder_input_text, torch.Tensor):
                assert len(decoder_input_text.shape) == 2
                labels = decoder_input_text.to(device)
            else:
                labels = torch.LongTensor(
                    self.decoder_tokenizer.encode(
                        [decoder_input_text],
                        bos=True,
                        eos=False,
                        output_type=yttm.OutputType.ID,
                    )
                ).to(device)
        else:
            labels = torch.LongTensor(
                [
                    [
                        self.decoder_tokenizer.bos_id,
                    ]
                ]
            ).to(device)

        args = {
            "get_logits": True,
            "return_encoder_state": True,
            "cached_encoder_state": None,
            "labels": labels,
        }
        if second_input_text is not None:
            args = {
                **args,
                "first_encoder_input_ids": encoder_input_ids,
                "second_encoder_input_ids": second_encoder_input_ids,
            }
        else:
            args = {
                **args,
                "input_ids": encoder_input_ids,
            }

        get_cached_encoder_state = False
        while args["labels"].shape[-1] < max_length:
            model_out = self.model(**args)
            logits = model_out["lm_probs"][:, -1, :]
            next_token = self.sample(logits, temperatype, **kwargs)
            if not get_cached_encoder_state:
                args["cached_encoder_state"] = model_out["encoder_hidden_state"]
                args["return_encoder_state"] = False
                get_cached_encoder_state = True
            args["labels"] = torch.cat([args["labels"], next_token], dim=-1)
        generated_seq = self.decoder_tokenizer.decode(
            args["labels"].cpu().tolist()[0],
            ignore_ids=(
                self.decoder_tokenizer.pad_id,
                self.decoder_tokenizer.unk_id,
                self.decoder_tokenizer.bos_id,
                self.decoder_tokenizer.eos_id,
            ),
        )[0]

        return generated_seq
