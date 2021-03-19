import torch
import torch.nn as nn
import torch.nn.functional as F

from common_layers import FFN, MultiHeadAttention, TransformerEmbedding


class PreLNEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model=512,
            n_heads=8,
            dim_feedforward=2048,
            dropout=0.1
        ):
        super(PreLNEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FFN(d_model, dim_feedforward, dropout)
        self.pre_attn_ln = nn.LayerNorm(d_model)
        self.pre_ffn_ln = nn.LayerNorm(d_model)

    def forward(
            self,
            hidden,
            attention_mask=None,
            get_attention_scores=False
        ):
        attn_scores = None
        block_state = self.pre_attn_ln(hidden)
        block_state = self.self_attention(
            query=block_state,
            key=block_state,
            value=block_state,
            attention_mask=attention_mask,
            get_attention_scores=get_attention_scores
        )
        if get_attention_scores:
            attn_scores = block_state[1]
        block_state = block_state[0]
        block_state = block_state + hidden

        ffn_block_state = self.pre_ffn_ln(block_state)
        ffn_block_state = self.ffn(ffn_block_state)
        ffn_block_state = ffn_block_state + block_state

        output = (ffn_block_state, )
        if get_attention_scores:
            output = output + (attn_scores, )
        return output


class PreLNEncoder(nn.Module):
    def __init__(
            self,
            num_layers,
            **kwargs
        ):
        super(PreLNEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList([
            PreLNEncoderLayer(**kwargs) for _ in range(num_layers)
        ])

    def forward(
            self,
            hidden,
            attention_mask=None,
            get_attention_scores=False
        ):
        attn_scores = []
        for layer in self.encoder_layers:
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                get_attention_scores=get_attention_scores
            )
            if get_attention_scores:
                attn_scores.append(hidden[1])
            hidden = hidden[0]
        output = (hidden, )
        if get_attention_scores:
            attn_scores = torch.stack(attn_scores, dim=-1)
            output = output + (attn_scores, )
        return output


class PreLNTransformerEncoder(nn.Module):
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
            use_layer_norm=False,
            pos_embedding_type='embedding'
        ):
        super(PreLNTransformerEncoder, self).__init__()
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_length=max_length,
            pad_token_id=pad_token_id,
            token_type_vocab_size=token_type_vocab_size,
            pos_embedding_type=pos_embedding_type,
            dropout=dropout,
            use_layer_norm=use_layer_norm
        )
        self.encoder = PreLNEncoder(
            num_layers=num_layers,
            d_model=d_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.post_encoder_ln = nn.LayerNorm(d_model)

    def create_attention_mask(self, attention_mask, input_shape, device):
        # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
        if attention_mask is None:
            attention_mask = torch.ones(*input_shape, device=device)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        attention_mask = (1.0 - attention_mask) * -10000.0
        return attention_mask

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            get_attention_scores=False
        ):

        attention_mask = self.create_attention_mask(
            attention_mask,
            input_ids.shape,
            input_ids.device
        )
        embeddings = self.embedding(input_ids, token_type_ids)
        hidden = self.encoder(
            embeddings,
            attention_mask=attention_mask,
            get_attention_scores=get_attention_scores
        )
        hidden_ln = self.post_encoder_ln(hidden[0])
        output = (hidden_ln, )
        if get_attention_scores:
            output = output + (hidden[1], )
        return output
