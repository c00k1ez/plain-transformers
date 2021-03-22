import torch


def create_attention_mask(attention_mask, input_shape, device):
    # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
    if attention_mask is None:
        attention_mask = torch.ones(*input_shape, device=device)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
    attention_mask = (1.0 - attention_mask) * -10000.0
    return attention_mask
