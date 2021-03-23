import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: add more flexible opportunity to init encoder & decoder
# TODO: add label smoothing loss
class Transformer(nn.Module):
    def __init__(
        self,
        encoder_class,
        decoder_class,
        encoder_vocab_size,
        decoder_vocab_size,
        use_token_type_embeddings=False,
        share_decoder_head_weights=True,
        share_encoder_decoder_embeddings=False,
        **kwargs
    ):
        super(Transformer, self).__init__()
        self.encoder = encoder_class(
            **kwargs,
            use_token_type_embeddings=use_token_type_embeddings,
            vocab_size=encoder_vocab_size
        )
        self.decoder = decoder_class(**kwargs, vocab_size=decoder_vocab_size)
        self.lm_head = nn.Linear(kwargs['d_model'], decoder_vocab_size, bias=False)
        if share_decoder_head_weights:
            self.lm_head.weight = self.decoder.embedding.token_embedding.weight
        if share_encoder_decoder_embeddings:
            self.encoder.embedding.token_embedding.weight = self.decoder.embedding.token_embedding.weight
        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=kwargs['pad_token_id']
        )
        self.pad_token_id = kwargs['pad_token_id']

    def forward(
        self,
        input_ids,
        labels,
        token_type_ids=None,
        decoder_attention_mask=None,
        encoder_attention_mask=None,
        get_attention_scores=False,
        cached_encoder_state=None,
        return_encoder_state=False,
        compute_loss=False
    ):
        encoder_state = None
        attn_scores = {}
        if cached_encoder_state is not None:
            encoder_state = cached_encoder_state
        else:
            encoder_state = self.encoder(
                input_ids=input_ids,
                attention_mask=encoder_attention_mask,
                token_type_ids=token_type_ids,
                get_attention_scores=get_attention_scores
            )
            if get_attention_scores:
                attn_scores['encoder'] = encoder_state[1]

            encoder_state = {'key': encoder_state[0], 'value': encoder_state[0]}
        hidden = self.decoder(
            input_ids=labels,
            encoder_hidden_state=encoder_state,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            get_attention_scores=get_attention_scores
        )
        if get_attention_scores:
            attn_scores['decoder'] = hidden[1]
        hidden = hidden[0]
        raw_probs = self.lm_head(hidden)

        output = {
            'lm_probs': torch.softmax(raw_probs, dim=-1)
        }
        if return_encoder_state:
            output['encoder_hidden_state'] = encoder_state
        if compute_loss:
            batch_size = labels.shape[0]
            labels = torch.cat([
                labels[:, 1:],
                torch.LongTensor([[self.pad_token_id]], device=labels.device).repeat(batch_size, 1)
            ], dim=-1)
            output['loss_val'] = self.loss_function(raw_probs.permute(0, 2, 1), labels)
        return output
