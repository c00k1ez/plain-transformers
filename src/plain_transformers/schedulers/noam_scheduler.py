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

from typing import List, Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler


class NoamScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: Optional[int] = 4000,
    ) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(NoamScheduler, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        last_step = max(1, self.last_epoch)
        scale_factor = self.d_model ** (-0.5) * min(last_step ** (-0.5), last_step * self.warmup_steps ** (-1.5))
        new_lrs = [lr * scale_factor for lr in self.base_lrs]
        return new_lrs
