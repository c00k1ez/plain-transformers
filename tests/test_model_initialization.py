import pytest
import torch
import torch.nn.functional as F

from src.plain_transformers.initializations import initialize_weights, normal_initialization
from src.plain_transformers.layers import PostLNTransformerEncoder


def test_initialize_weights():
    torch.manual_seed(42)
    model_config = {
        "d_model": 128,
        "vocab_size": 500,
        "max_length": 100,
        "pad_token_id": 0,
        "token_type_vocab_size": 0,
        "n_heads": 4,
        "dim_feedforward": 200,
        "num_layers": 1,
    }
    model_one = PostLNTransformerEncoder(**model_config)
    model_two = PostLNTransformerEncoder(**model_config)

    layer_shape = model_one.encoder.encoder_layers[0].ffn.layer_inc.bias.data.shape
    assert not torch.equal(model_one.encoder.encoder_layers[0].ffn.layer_inc.bias.data, torch.zeros(layer_shape))

    initialize_weights(model_one, normal_initialization, init_range=0.02)
    assert torch.equal(model_one.encoder.encoder_layers[0].ffn.layer_inc.bias.data, torch.zeros(layer_shape))
    model_two.apply(normal_initialization)
    assert torch.equal(model_two.encoder.encoder_layers[0].ffn.layer_inc.bias.data, torch.zeros(layer_shape))
