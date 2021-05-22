import pytest
import torch
import torch.nn.functional as F

from src.plain_transformers.losses import UnlikelihoodLoss


class TestUnlikelihoodLoss:
    @pytest.mark.parametrize("reduction", ("sum", "mean", "none"))
    @pytest.mark.parametrize("context_type", ("sentence", "full_context"))
    def test_loss_shapes(self, reduction, context_type):
        torch.manual_seed(42)
        pad_id = 0
        criterion = UnlikelihoodLoss(0.25, ignore_index=pad_id, context_type=context_type, reduction=reduction)
        batch_size, seq_len, vocab_size = 5, 10, 60
        input = torch.rand((batch_size, seq_len, vocab_size))
        target = torch.randint(1, vocab_size, (batch_size, seq_len))
        # add paddings
        target[:, -1] = torch.LongTensor(
            [
                pad_id,
            ]
            * target.shape[0]
        )
        with torch.no_grad():
            loss_val = criterion(input, target)
        if reduction == "none":
            assert tuple(loss_val.shape) == (batch_size * seq_len,), tuple(loss_val.shape)
        else:
            assert len(loss_val.shape) == 0 and not loss_val.isnan()
