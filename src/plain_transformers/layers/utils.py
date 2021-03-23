from typing import Tuple, Callable

import torch
import torch.nn.functional as F


def create_attention_mask(
    attention_mask: torch.Tensor,
    input_shape: Tuple[int, int],
    device: torch.device,
    src_size: int = 1,
) -> torch.Tensor:
    # [batch_size, seq_len] -> [batch_size, 1, tgt_size, seq_len]
    if attention_mask is None:
        attention_mask = torch.ones(*input_shape, device=device)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
    if src_size > 1:
        attention_mask = attention_mask.repeat(1, 1, src_size, 1)
    attention_mask = (1.0 - attention_mask) * -10000.0
    return attention_mask


def act_to_func(act_name: str) -> Callable:
    acts = {"gelu": F.gelu, "relu": F.relu}
    if act_name in acts:
        return acts[act_name]
    else:
        return F.relu
