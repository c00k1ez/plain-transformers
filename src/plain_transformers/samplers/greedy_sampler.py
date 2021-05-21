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


class GreedySampler(BaseSampler):
    def __init__(self, *args, **kwargs) -> None:
        super(GreedySampler, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def sample(self, logits: torch.Tensor, temperatype: float, **kwargs) -> torch.Tensor:
        probs = torch.softmax(logits / temperatype, dim=-1)
        max_val = probs.max(dim=-1)[1].unsqueeze(-1)
        return max_val
