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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnlikelihoodLoss(nn.Module):
    def __init__(
        self,
        alpha: float,
        ignore_index: Optional[int] = 0,
        context_type: Optional[str] = "full_context",
        custom_likelihood_loss: Optional[nn.Module] = None,
        reduction: Optional[str] = "mean",
    ) -> None:
        super(UnlikelihoodLoss, self).__init__()
        assert context_type in ["full_context", "sentence"]
        self.context_type = context_type
        assert ignore_index >= 0
        self.ignore_index = ignore_index
        assert reduction in ["sum", "mean", "none"]
        self.reduction = reduction
        self.alpha = alpha
        if custom_likelihood_loss is not None:
            self.custom_likelihood_loss = custom_likelihood_loss
            assert custom_likelihood_loss.ignore_index == ignore_index
            assert custom_likelihood_loss.reduction == reduction
        else:
            self.custom_likelihood_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.context_type == "sentence" and len(input.shape) > 2:
            raise ValueError(
                "If you use context_type 'sentence', you should pass 3-d input tensor and 2-d target tensor"
            )
        raw_logits = F.log_softmax(input, dim=-1)
        unchanged_target = target.clone()
        with torch.no_grad():
            if self.context_type == "full_context":
                target = target.view(-1)
                raw_logits = raw_logits.view(-1, raw_logits.size(-1))
                negative_candidates = target.unsqueeze(0).expand(target.size(0), -1)
                mask = torch.empty_like(negative_candidates).fill_(self.ignore_index).triu()
                negative_candidates = negative_candidates.tril(-1) + mask
            elif self.context_type == "sentence":
                negative_candidates = target.unsqueeze(1).expand(-1, target.size(1), -1)
                mask = torch.empty_like(negative_candidates).fill_(self.ignore_index).triu()
                negative_candidates = negative_candidates.tril(-1) + mask

            negative_candidates.masked_fill_(negative_candidates == target.unsqueeze(-1), self.ignore_index)
            negative_targets = torch.zeros_like(raw_logits).scatter_(-1, negative_candidates, 1)
            # delete paddings from negative_targets
            negative_targets.masked_fill_(target.unsqueeze(-1) == self.ignore_index, 0)
            negative_targets[..., self.ignore_index] = 0.0

        # calculate unlikelihood part
        inverse_probs = torch.clamp((1.0 - raw_logits.exp()), min=1e-5)
        ul_loss = -torch.log(inverse_probs) * negative_targets
        if len(ul_loss.shape) > 2:
            ul_loss = ul_loss.view(-1, ul_loss.shape[-1])
        ul_loss = ul_loss.sum(dim=-1)

        # calculate mle loss
        mle_loss = self.custom_likelihood_loss(input.view(-1, input.shape[-1]), unchanged_target.view(-1))
        if self.reduction == "sum":
            ul_loss = ul_loss.sum()
        elif self.reduction == "mean":
            ul_loss = ul_loss.sum() / (target.view(-1) != self.ignore_index).sum()
        assert mle_loss.shape == ul_loss.shape
        return self.alpha * ul_loss + mle_loss
