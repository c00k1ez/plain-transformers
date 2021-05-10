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
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        self.smooth = smoothing
        self.ignore_index = ignore_index

    def forward(self, input, target, reduction="mean"):
        assert reduction in ["mean", "sum"]
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.shape) * self.smooth / (input.shape[-1] - 2)
        weight.scatter_(-1, target.unsqueeze(-1), (1.0 - self.smooth))
        weight.masked_fill_((target == 1).unsqueeze(-1), 0)
        loss = (-weight * log_prob).sum(dim=-1).sum()
        if reduction == "mean":
            loss = loss / (target != 1).sum()
        return loss
