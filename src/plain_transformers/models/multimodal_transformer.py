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
        self.lm_head = nn.Linear(
            kwargs["d_model"], decoder_vocab_size, bias=False
        )
        if share_decoder_head_weights:
            self.lm_head.weight = self.decoder.embedding.token_embedding.weight
        if share_encoder_decoder_embeddings:
            self.first_encoder.embedding.token_embedding.weight = (
                self.decoder.embedding.token_embedding.weight
            )
            self.second_encoder.embedding.token_embedding.weight = (
                self.decoder.embedding.token_embedding.weight
            )
        if share_encoder_embeddings:
            self.first_encoder.embedding.token_embedding.weight = (
                self.second_encoder.embedding.token_embedding.weight
            )
        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=kwargs["pad_token_id"]
        )
        self.pad_token_id = kwargs["pad_token_id"]

    def forward(
        self,
        first_encoder_input_ids,
        second_encoder_input_ids,
        labels,
        decoder_attention_mask=None,
        fisrt_encoder_attention_mask=None,
        second_encoder_attention_mask=None,
        get_attention_scores=False,
        cached_encoder_state=None,
        return_encoder_state=False,
        compute_loss=False,
    ):
        first_encoder_state, second_encoder_state = None, None
        attn_scores = {}
        if cached_encoder_state is not None:
            first_encoder_state, second_encoder_state = cached_encoder_state
        else:
            first_encoder_state = self.first_encoder(
                input_ids=first_encoder_input_ids,
                attention_mask=fisrt_encoder_attention_mask,
                token_type_ids=None,
                get_attention_scores=get_attention_scores,
            )

            second_encoder_state = self.second_encoder(
                input_ids=second_encoder_input_ids,
                attention_mask=second_encoder_attention_mask,
                token_type_ids=None,
                get_attention_scores=get_attention_scores,
            )
            if get_attention_scores:
                attn_scores["encoder"] = {
                    "first_encoder": first_encoder_state[1],
                    "second_encoder": second_encoder_state[1],
                }

            first_encoder_state = {
                "key": first_encoder_state[0],
                "value": first_encoder_state[0],
            }
            second_encoder_state = {
                "key": second_encoder_state[0],
                "value": second_encoder_state[0],
            }
        hidden = self.decoder(
            input_ids=labels,
            first_encoder_hidden_state=first_encoder_state,
            second_encoder_hidden_state=second_encoder_state,
            attention_mask=decoder_attention_mask,
            fisrt_encoder_attention_mask=fisrt_encoder_attention_mask,
            second_encoder_attention_mask=second_encoder_attention_mask,
            get_attention_scores=get_attention_scores,
        )
        if get_attention_scores:
            attn_scores["decoder"] = hidden[1]
        hidden = hidden[0]
        raw_probs = self.lm_head(hidden)

        output = {"lm_probs": torch.softmax(raw_probs, dim=-1)}
        if return_encoder_state:
            output["encoder_hidden_state"] = (
                first_encoder_state,
                second_encoder_state,
            )
        if compute_loss:
            batch_size = labels.shape[0]
            labels = torch.cat(
                [
                    labels[:, 1:],
                    torch.LongTensor(
                        [[self.pad_token_id]], device=labels.device
                    ).repeat(batch_size, 1),
                ],
                dim=-1,
            )
            output["loss_val"] = self.loss_function(
                raw_probs.permute(0, 2, 1), labels
            )
        return output
