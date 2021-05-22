import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.plain_transformers.losses import LabelSmoothingLoss


class TestLabelSmoothingLoss:
    @pytest.mark.parametrize("reduction", ("sum", "mean", "none"))
    def test_loss_without_smoothing(self, reduction):
        pad_id = 42
        torch.manual_seed(42)
        input = torch.rand((5, 10, 123))
        target = torch.randint(123, (5, 10))
        original_criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction=reduction)
        implemented_criterion = LabelSmoothingLoss(ignore_index=pad_id, reduction=reduction)
        orig_output = original_criterion(input.view(-1, 123), target.view(-1))
        imp_output = implemented_criterion(input.view(-1, 123), target.view(-1))
        assert torch.allclose(orig_output, imp_output)
