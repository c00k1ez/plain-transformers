import pytest
import torch
import torch.nn.functional as F

from src.plain_transformers.layers.utils import act_to_func, create_attention_mask


class TestUtils:
    # just simple & useless test
    def test_act_to_func(self):
        func_names = {
            "gelu": F.gelu,
            "relu": F.relu,
            "relu6": F.relu6,
            "elu": F.elu,
            "selu": F.selu,
            "celu": F.celu,
            "leaky_relu": F.leaky_relu,
            "tanh": F.tanh,
        }
        tensor = torch.rand((5, 5))
        for name in func_names:
            assert torch.equal(func_names[name](tensor), act_to_func(name)(tensor))
        assert torch.equal(act_to_func("not_implemented_func")(tensor), F.relu(tensor))

    def test_create_attention_mask(self):
        attn_mask = torch.LongTensor([[1, 0, 0], [1, 1, 0]])
        device = torch.device("cpu")
        # [batch_size, seq_len] -> [batch_size, 1, src_size, seq_len]
        extended_mask = create_attention_mask(
            attention_mask=attn_mask,
            input_shape=attn_mask.shape,
            device=device,
            src_size=4,
        )
        ground_truth = torch.FloatTensor(
            [
                [
                    [
                        [-0.0, -10000.0, -10000.0],
                        [-0.0, -10000.0, -10000.0],
                        [-0.0, -10000.0, -10000.0],
                        [-0.0, -10000.0, -10000.0],
                    ]
                ],
                [
                    [
                        [-0.0, -0.0, -10000.0],
                        [-0.0, -0.0, -10000.0],
                        [-0.0, -0.0, -10000.0],
                        [-0.0, -0.0, -10000.0],
                    ]
                ],
            ]
        )
        assert torch.equal(extended_mask, ground_truth)

        extended_mask = create_attention_mask(
            attention_mask=None,
            input_shape=[2, 3],
            device=device,
            src_size=4,
        )
        ground_truth = torch.FloatTensor(
            [
                [
                    [
                        [-0.0, -0.0, -0.0],
                        [-0.0, -0.0, -0.0],
                        [-0.0, -0.0, -0.0],
                        [-0.0, -0.0, -0.0],
                    ]
                ],
                [
                    [
                        [-0.0, -0.0, -0.0],
                        [-0.0, -0.0, -0.0],
                        [-0.0, -0.0, -0.0],
                        [-0.0, -0.0, -0.0],
                    ]
                ],
            ]
        )

        assert torch.equal(extended_mask, ground_truth)
