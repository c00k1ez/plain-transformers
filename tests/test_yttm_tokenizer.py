import os

import pytest
import torch
import youtokentome as yttm

from src.plain_transformers import BPEWrapper


class TestBPEWrapper:
    def test_bpe_wrapper(self):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        files_directory = os.path.join(current_directory, "test_data")
        model = BPEWrapper(model=f"{files_directory}/test_yttm_tokenizer.model")
        sample = "hello world"
        encoded = model.encode(
            [
                sample,
            ],
            bos=True,
            eos=True,
            output_type=yttm.OutputType.SUBWORD,
        )[0]
        assert encoded == ["<BOS>", "▁h", "ell", "o", "▁", "w", "or", "l", "d", "<EOS>"]
