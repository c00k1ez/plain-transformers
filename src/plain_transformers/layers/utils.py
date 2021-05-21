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

from typing import Callable, Tuple, Union

import torch
import torch.nn.functional as F


def create_attention_mask(
    attention_mask: Union[torch.Tensor, None],
    input_shape: Tuple[int, int],
    device: torch.device,
    src_size: int = 1,
) -> torch.Tensor:
    # [batch_size, seq_len] -> [batch_size, 1, src_size, seq_len]
    if attention_mask is None:
        attention_mask = torch.ones(*input_shape).to(device)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
    if src_size > 1:
        attention_mask = attention_mask.repeat(1, 1, src_size, 1)
    attention_mask = (1.0 - attention_mask) * -10000.0
    return attention_mask


def act_to_func(act_name: str) -> Callable:
    acts = {
        "gelu": F.gelu,
        "relu": F.relu,
        "relu6": F.relu6,
        "elu": F.elu,
        "selu": F.selu,
        "celu": F.celu,
        "leaky_relu": F.leaky_relu,
        "tanh": F.tanh,
    }
    if act_name in acts:
        return acts[act_name]
    else:
        return F.relu
