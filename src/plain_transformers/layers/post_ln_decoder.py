import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_layers import (
    FFN,
    MultiHeadAttention,
    TransformerEmbedding,
    TransformerDecoder,
)
from .utils import create_attention_mask


class PostLNDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=512,
        n_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation_name="gelu",
        ln_eps=1e-12,
        context_len=512,
    ):
        super(PostLNDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            context_len=context_len,
        )
        self.self_attn_merge_matrix = nn.Linear(d_model, d_model)
        self.post_attn_ln = nn.LayerNorm(d_model, eps=ln_eps)

        self.cross_attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.cross_attn_merge_matrix = nn.Linear(d_model, d_model)
        self.post_cross_attn_ln = nn.LayerNorm(d_model, eps=ln_eps)

        self.ffn = FFN(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation_name=activation_name,
        )
        self.post_ffn_ln = nn.LayerNorm(d_model, eps=ln_eps)

    def forward(
        self,
        hidden,
        encoder_hidden_state,
        attention_mask=None,
        encoder_attention_mask=None,
        get_attention_scores=False,
    ):
        attn_scores = []
        self_attn_block = self.self_attention(
            query=hidden,
            key=hidden,
            value=hidden,
            attention_mask=attention_mask,
            get_attention_scores=get_attention_scores,
        )
        if get_attention_scores:
            attn_scores = [
                self_attn_block[1],
            ]
        self_attn_block = self_attn_block[0]
        self_attn_block = self.self_attn_merge_matrix(self_attn_block)
        self_attn_block = self_attn_block + hidden
        self_attn_block = self.post_attn_ln(self_attn_block)

        cross_attn_block = self.cross_attention(
            query=self_attn_block,
            key=encoder_hidden_state["key"],
            value=encoder_hidden_state["value"],
            attention_mask=encoder_attention_mask,
            get_attention_scores=get_attention_scores,
        )
        if get_attention_scores:
            attn_scores.append(cross_attn_block[1])
            attn_scores = torch.stack(attn_scores, dim=-1)
        cross_attn_block = cross_attn_block[0]
        cross_attn_block = self.cross_attn_merge_matrix(cross_attn_block)
        cross_attn_block = cross_attn_block + self_attn_block
        cross_attn_block = self.post_cross_attn_ln(cross_attn_block)

        ffn_block = self.ffn(cross_attn_block)
        ffn_block = ffn_block + cross_attn_block
        ffn_block = self.post_ffn_ln(ffn_block)

        output = (ffn_block,)
        if get_attention_scores:
            output = output + (attn_scores,)
        return output


class PostLNTransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        vocab_size,
        max_length,
        pad_token_id,
        token_type_vocab_size,
        dropout,
        n_heads,
        dim_feedforward,
        num_layers,
        use_embedding_layer_norm=False,
        pos_embedding_type="embedding",
        activation_name="gelu",
        ln_eps=1e-12,
    ):
        super(PostLNTransformerDecoder, self).__init__()
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_length=max_length,
            pad_token_id=pad_token_id,
            token_type_vocab_size=token_type_vocab_size,
            dropout=dropout,
            use_layer_norm=use_embedding_layer_norm,
            use_token_type_embeddings=False,
            ln_eps=ln_eps,
            pos_embedding_type=pos_embedding_type,
        )

        self.decoder = TransformerDecoder(
            num_layers=num_layers,
            decoder_class=PostLNDecoderLayer,
            d_model=d_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation_name=activation_name,
            ln_eps=ln_eps,
            context_len=max_length,
        )

    def forward(
        self,
        input_ids,
        encoder_hidden_state,
        attention_mask=None,
        encoder_attention_mask=None,
        get_attention_scores=False,
    ):
        attention_mask = create_attention_mask(
            attention_mask=attention_mask,
            input_shape=input_ids.shape,
            device=input_ids.device,
        )

        encoder_attention_mask = create_attention_mask(
            attention_mask=encoder_attention_mask,
            input_shape=encoder_hidden_state["key"].shape[:-1],
            device=encoder_hidden_state["key"].device,
            src_size=input_ids.shape[-1],
        )

        embeddings = self.embedding(input_ids)

        hidden = self.decoder(
            hidden=embeddings,
            encoder_hidden_state=encoder_hidden_state,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            get_attention_scores=get_attention_scores,
        )

        return hidden


class PostLNMultimodalDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=512,
        n_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation_name="gelu",
        ln_eps=1e-12,
        context_len=512,
    ):
        super(PostLNMultimodalDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            context_len=context_len,
        )
        self.self_attn_merge_matrix = nn.Linear(d_model, d_model)
        self.post_attn_ln = nn.LayerNorm(d_model, eps=ln_eps)

        self.first_cross_attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.first_cross_attn_merge_matrix = nn.Linear(d_model, d_model)
        self.first_post_cross_attn_ln = nn.LayerNorm(d_model, eps=ln_eps)

        self.second_cross_attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.second_cross_attn_merge_matrix = nn.Linear(d_model, d_model)
        self.second_post_cross_attn_ln = nn.LayerNorm(d_model, eps=ln_eps)

        self.ffn = FFN(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation_name=activation_name,
        )
        self.post_ffn_ln = nn.LayerNorm(d_model, eps=ln_eps)

    def forward(
        self,
        hidden,
        encoder_hidden_state,
        attention_mask=None,
        encoder_attention_mask=None,
        get_attention_scores=False,
    ):
        (first_encoder_hidden_state, second_encoder_hidden_state) = (
            encoder_hidden_state[0],
            encoder_hidden_state[1],
        )
        (first_encoder_attention_mask, second_encoder_attention_mask) = (
            encoder_attention_mask[0],
            encoder_attention_mask[1],
        )
        attn_scores = []
        self_attn_block = self.self_attention(
            query=hidden,
            key=hidden,
            value=hidden,
            attention_mask=attention_mask,
            get_attention_scores=get_attention_scores,
        )
        if get_attention_scores:
            attn_scores = [
                self_attn_block[1],
            ]
        self_attn_block = self_attn_block[0]
        self_attn_block = self.self_attn_merge_matrix(self_attn_block)
        self_attn_block = self_attn_block + hidden
        self_attn_block = self.post_attn_ln(self_attn_block)

        first_cross_attn_block = self.first_cross_attention(
            query=self_attn_block,
            key=first_encoder_hidden_state["key"],
            value=first_encoder_hidden_state["value"],
            attention_mask=first_encoder_attention_mask,
            get_attention_scores=get_attention_scores,
        )
        if get_attention_scores:
            attn_scores.append(first_cross_attn_block[1])

        first_cross_attn_block = first_cross_attn_block[0]
        first_cross_attn_block = self.first_cross_attn_merge_matrix(
            first_cross_attn_block
        )
        first_cross_attn_block = first_cross_attn_block + self_attn_block
        first_cross_attn_block = self.first_post_cross_attn_ln(
            first_cross_attn_block
        )

        second_cross_attn_block = self.second_cross_attention(
            query=first_cross_attn_block,
            key=second_encoder_hidden_state["key"],
            value=second_encoder_hidden_state["value"],
            attention_mask=second_encoder_attention_mask,
            get_attention_scores=get_attention_scores,
        )
        if get_attention_scores:
            attn_scores.append(second_cross_attn_block[1])
            attn_scores = torch.stack(attn_scores, dim=-1)
        second_cross_attn_block = second_cross_attn_block[0]
        second_cross_attn_block = self.second_cross_attn_merge_matrix(
            second_cross_attn_block
        )
        second_cross_attn_block = (
            second_cross_attn_block + first_cross_attn_block
        )
        second_cross_attn_block = self.second_post_cross_attn_ln(
            second_cross_attn_block
        )

        ffn_block = self.ffn(second_cross_attn_block)
        ffn_block = ffn_block + second_cross_attn_block
        ffn_block = self.post_ffn_ln(ffn_block)

        output = (ffn_block,)
        if get_attention_scores:
            output = output + (attn_scores,)
        return output


class PostLNMultimodalTransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        vocab_size,
        max_length,
        pad_token_id,
        token_type_vocab_size,
        dropout,
        n_heads,
        dim_feedforward,
        num_layers,
        use_embedding_layer_norm=False,
        pos_embedding_type="embedding",
        activation_name="gelu",
        ln_eps=1e-12,
    ):
        super(PostLNMultimodalTransformerDecoder, self).__init__()
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_length=max_length,
            pad_token_id=pad_token_id,
            token_type_vocab_size=token_type_vocab_size,
            dropout=dropout,
            use_layer_norm=use_embedding_layer_norm,
            use_token_type_embeddings=False,
            ln_eps=ln_eps,
            pos_embedding_type=pos_embedding_type,
        )

        self.decoder = TransformerDecoder(
            num_layers=num_layers,
            decoder_class=PostLNMultimodalDecoderLayer,
            d_model=d_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation_name=activation_name,
            ln_eps=ln_eps,
            context_len=max_length,
        )

    def forward(
        self,
        input_ids,
        first_encoder_hidden_state,
        second_encoder_hidden_state,
        attention_mask=None,
        first_encoder_attention_mask=None,
        second_encoder_attention_mask=None,
        get_attention_scores=False,
    ):
        attention_mask = create_attention_mask(
            attention_mask=attention_mask,
            input_shape=input_ids.shape,
            device=input_ids.device,
        )

        first_encoder_attention_mask = create_attention_mask(
            attention_mask=first_encoder_attention_mask,
            input_shape=first_encoder_hidden_state["key"].shape[:-1],
            device=first_encoder_hidden_state["key"].device,
            src_size=input_ids.shape[-1],
        )

        second_encoder_hidden_state = create_attention_mask(
            attention_mask=second_encoder_attention_mask,
            input_shape=second_encoder_hidden_state["key"].shape[:-1],
            device=second_encoder_hidden_state["key"].device,
            src_size=input_ids.shape[-1],
        )

        embeddings = self.embedding(input_ids)

        hidden = self.decoder(
            hidden=embeddings,
            encoder_hidden_state=(
                first_encoder_hidden_state,
                second_encoder_hidden_state,
            ),
            attention_mask=attention_mask,
            encoder_attention_mask=(
                first_encoder_attention_mask,
                second_encoder_attention_mask,
            ),
            get_attention_scores=get_attention_scores,
        )

        return hidden
