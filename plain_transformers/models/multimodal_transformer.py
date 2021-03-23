import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: add more flexible opportunity to init encoder & decoder
# TODO: add label smoothing loss
# TODO: write more complex solution for embedding sharing
class MultimodalTransformer(nn.Module):
    def __init__(
            self,
            first_encoder_class,
            second_encoder_class,
            decoder_class,
            first_encoder_vocab_size,
            second_encoder_vocab_size,
            decoder_vocab_size,
            use_token_type_embeddings=False,
            share_decoder_head_weights=True,
            share_encoder_decoder_embeddings=False,
            share_encoder_embeddings=False,
            **kwargs
        ):
        super(MultimodalTransformer, self).__init__()
        self.first_encoder = first_encoder_class(
            **kwargs,
            use_token_type_embeddings=use_token_type_embeddings,
            vocab_size=first_encoder_vocab_size
        )
        self.second_encoder = second_encoder_class(
            **kwargs,
            use_token_type_embeddings=use_token_type_embeddings,
            vocab_size=second_encoder_vocab_size
        )

        self.decoder = decoder_class(**kwargs, vocab_size=decoder_vocab_size)
        self.lm_head = nn.Linear(kwargs['d_model'], decoder_vocab_size, bias=False)
        if share_decoder_head_weights:
            self.lm_head.weight = self.decoder.embedding.token_embedding.weight
        if share_encoder_decoder_embeddings:
            self.first_encoder.embedding.token_embedding.weight = self.decoder.embedding.token_embedding.weight
            self.second_encoder.embedding.token_embedding.weight = self.decoder.embedding.token_embedding.weight
        if share_encoder_embeddings:
            self.first_encoder.embedding.token_embedding.weight = self.second_encoder.embedding.token_embedding.weight
        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=kwargs['pad_token_id']
        )
        self.pad_token_id = kwargs['pad_token_id']

    def forward(
            self,
            first_encoder_input_ids,
            second_encoder_input_ids,
            labels,
            token_type_ids=None,
            decoder_attention_mask=None,
            fisrt_encoder_attention_mask=None,
            second_encoder_attention_mask=None,
            get_attention_scores=False,
            cached_encoder_state=None,
            return_encoder_state=False,
            compute_loss=False
        ):
        pass