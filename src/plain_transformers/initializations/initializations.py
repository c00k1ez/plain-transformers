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

from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn


def initialize_weights(
    model: nn.Module,
    initialization_function: Callable,
    pre_initialization: Optional[Callable] = None,
    is_admin=False,
    **kwargs,
) -> None:
    pre_init_outputs = {}
    if pre_initialization is not None:
        pre_init_outputs = pre_initialization(model)  # must be dict
    initialization_function_with_args = partial(initialization_function, **kwargs, **pre_init_outputs)
    model.apply(initialization_function_with_args)


def normal_initialization(module: nn.Module, init_range: Optional[float] = 0.02) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=init_range)
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        if module.bias is not None:
            module.bias.data.zero_()
        module.weight.data.fill_(1.0)
