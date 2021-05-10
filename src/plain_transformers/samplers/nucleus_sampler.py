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

import torch

from .base_sampler import BaseSampler


class NucleusSampler(BaseSampler):
    """
    Sampling based on
    https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """

    def __init__(self, *args, **kwargs) -> None:
        super(NucleusSampler, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def sample(self, logits: torch.Tensor, temperatype: float, **kwargs) -> torch.Tensor:
        # logits shape: [1, vocab_size]
        top_k, top_p = logits.shape[-1], 0.0
        if "top_k" in kwargs:
            top_k = kwargs["top_k"]
        if "top_p" in kwargs:
            top_p = kwargs["top_p"]
        logits = logits / temperatype
        if top_k > 0:
            inds_to_delete = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[inds_to_delete] = float("-inf")

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs >= top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = torch.zeros_like(logits, dtype=sorted_indices_to_remove.dtype).scatter_(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )  # .to(logits.device)
            logits[indices_to_remove] = float("-inf")

        return torch.multinomial(torch.softmax(logits, dim=-1), 1)
