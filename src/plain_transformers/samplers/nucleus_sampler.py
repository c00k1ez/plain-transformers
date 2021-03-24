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
    def __init__(self, *args, **kwargs) -> None:
        super(NucleusSampler, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def sample(
        self, logits: torch.Tensor, temperatype: float, **kwargs
    ) -> torch.Tensor:
        top_k, top_p = None, None
        if "top_k" in kwargs:
            top_k = kwargs["top_k"]
        if "top_p" in kwargs:
            top_p = kwargs["top_p"]
        assert (top_k is None and top_p is not None) or (
            top_k is not None and top_p is None
        )
        probs = torch.softmax(logits / temperatype, dim=-1)
        top_vals_inds = None
        if top_k is not None:
            top_vals_inds = probs.topk(top_k)
            ind = torch.multinomial(top_vals_inds.values, 1)
            max_val = top_vals_inds.indices[0, ind]
        else:
            # TODO: implement tok_p sampling
            pass
        return max_val
