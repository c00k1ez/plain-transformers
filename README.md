# Simple way to use transformer models

[![PyPI version](https://badge.fury.io/py/plain-transformers.svg)](https://badge.fury.io/py/plain-transformers)
![Build Status](https://github.com/c00k1ez/plain_transformers/actions/workflows/ci_tests.yaml/badge.svg)

## Features
* Multimodal Transformer (text, text -> text; for example for Automatic Post-Editing task);
* Label smoothing loss;
* Unlikelihood loss (sentence-level and for full context), [arxiv](https://arxiv.org/abs/1908.04319);
* LayerDrop technique for transformer encoder/decoder layers, [arxiv](https://arxiv.org/abs/1909.11556);
* Pre/Post LayerNorm encoders (Decoders in process), [arxiv](https://arxiv.org/abs/2004.08249);
* ADMIN Initialization (in future), [arxiv](https://arxiv.org/abs/2004.08249);
* Top-k/Top-p sampling with temperature, [arxiv](https://arxiv.org/abs/1904.09751);

## Installation
```pip install plain-transformers```

## Usage
Multimodal transformer example with two tokenizers:

**Step one**: import model and some usefull staff;
```python
import torch

from plain_transformers.models import MultimodalTransformer
from plain_transformers.layers import MultimodalTransformerDecoder
from plain_transformers.layers import TransformerEncoder

from plain_transformers import BPEWrapper
from plain_transformers.initializations import normal_initialization, initialize_weights
from plain_transformers.samplers.nucleus_sampler import NucleusSampler

import youtokentome as yttm
```
**Step two**: train and load tokenizers;
```python
# train your encoder tokenizer
yttm.BPE.train(..., model='encoder_tokenizer.model')
# train your decoder tokenizer
yttm.BPE.train(..., model='decoder_tokenizer.model')

# load tokenizers
encoder_tokenizer = BPEWrapper(model='encoder_tokenizer.model')
decoder_tokenizer = BPEWrapper(model='decoder_tokenizer.model')
```
**Step three**: init out model configuration;
```python
cfg = {
    'd_model': 768,
    'first_encoder': {
        'first_encoder_vocab_size': encoder_tokenizer.vocab_size(),
        'first_encoder_max_length': 512,
        'first_encoder_pad_token_id': encoder_tokenizer.pad_id,
        'first_encoder_token_type_vocab_size': 2,
        'first_encoder_n_heads': 8,
        'first_encoder_dim_feedforward': 2048,
        'first_encoder_num_layers': 3,
        'first_encoder_type': 'post_ln'
    },
    'second_encoder': {
        'second_encoder_vocab_size': encoder_tokenizer.vocab_size(),
        'second_encoder_max_length': 512,
        'second_encoder_pad_token_id': encoder_tokenizer.pad_id,
        'second_encoder_token_type_vocab_size': 2,
        'second_encoder_n_heads': 8,
        'second_encoder_dim_feedforward': 2048,
        'second_encoder_num_layers': 3,
        'second_encoder_type': 'post_ln'
    },
    'decoder': {
        'decoder_max_length': 512,
        'decoder_vocab_size': decoder_tokenizer.vocab_size(),
        'decoder_pad_token_id': decoder_tokenizer.pad_id,
        'decoder_token_type_vocab_size': 2,
        'decoder_n_heads': 8,
        'decoder_dim_feedforward': 2048,
        'decoder_num_layers': 3,
        'decoder_type': 'post_ln'
    },
}
```
**Step four**: initialize model and apply weight initialisation (with default parameter ```std=0.02```);
```python
model = MultimodalTransformer(
    TransformerEncoder,
    TransformerEncoder,
    MultimodalTransformerDecoder,
    cfg['d_model'],
    **cfg['first_encoder'],
    **cfg['second_encoder'],
    **cfg['decoder'],
    share_decoder_head_weights=True,
    share_encoder_decoder_embeddings=False,
    share_encoder_embeddings=True,
)

initialize_weights(model, normal_initialization, init_range=0.02)
```
**Step five**: train our model like ordinary seq2seq;
```python
train(model, ...)
```
**Step six**: initialize Sampler and generate model answer;
```python
sampler = NucleusSampler(model, encoder_tokenizer=(encoder_tokenizer, encoder_tokenizer), decoder_tokenizer=decoder_tokenizer)
sampler.generate('Hello Bob, what are you doing?', second_input_text='Fine, thanks!', top_k=5)
```
## Example
You can find working example of NMT [here](https://colab.research.google.com/drive/1WA_CcmDD-O51foBvSOMKT4sZD-zuzQwA?usp=sharing).
