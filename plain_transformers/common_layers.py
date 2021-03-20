import torch
import torch.nn as nn
import torch.nn.functional as F


def act_to_func(act_name):
    acts = {
        "gelu": F.gelu,
        "relu": F.relu
    }
    if act_name in acts:
        return acts[act_name]
    else:
        return F.relu


class FFN(nn.Module):
    def __init__(
            self,
            d_model=512,
            dim_feedforward=2048,
            dropout=0.1,
            activation_name="gelu"
        ):
        super(FFN, self).__init__()   
        self.d_model = d_model
        self.activation_name = activation_name
        self.dim_feedforward = dim_feedforward
        self.layer_inc = nn.Linear(d_model, dim_feedforward)
        self.layer_reduce = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
   
    def forward(self, hidden):
        hid_state = self.layer_inc(hidden)
        hid_state = self.dropout(act_to_func(self.activation_name)(hid_state))
        return self.layer_reduce(hid_state)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model=512,
            n_heads=8,
            dropout=0.1,
            query_input_dim=None,
            key_input_dim=None,
            value_input_dim=None
        ):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.hidden_per_head = d_model // n_heads
        self.scale = self.hidden_per_head ** 0.5
        self.query_input_dim = d_model if query_input_dim is None else query_input_dim
        self.key_input_dim = d_model if key_input_dim is None else key_input_dim
        self.value_input_dim = d_model if value_input_dim is None else value_input_dim

        self.key_projection = nn.Linear(self.key_input_dim, d_model)
        self.query_projection = nn.Linear(self.query_input_dim, d_model)
        self.value_projection = nn.Linear(self.value_input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def _transpose_to_heads(self, x):
        # (batch_size, seq_len, emb_dim) -> (batch_size, seq_len, n_heads, hidden_per_head)
        new_shape = x.size()[:-1] + (self.n_heads, self.hidden_per_head)
        x = x.view(*new_shape)
        # (batch_size, seq_len, n_heads, hidden_per_head) ->
        # -> (batch_size, n_heads, seq_len, hidden_per_head)
        return x.permute(0, 2, 1, 3)

    def _generate_decoder_self_attn_mask(self):
        pass

    def forward(
            self,
            query,
            key,
            value,
            attention_mask=None,
            use_decoder_self_attn_mask=False,
            get_attention_scores=False
        ):
        query_proj = self.query_projection(query)
        key_proj = self.key_projection(key)
        value_proj = self.value_projection(value)

        query_proj = self._transpose_to_heads(query_proj)
        key_proj = self._transpose_to_heads(key_proj)
        value_proj = self._transpose_to_heads(value_proj)

        raw_scores = torch.matmul(query_proj, key_proj.transpose(-1, -2))

        if attention_mask is not None:
            raw_scores = raw_scores + attention_mask       
        attn_scores = F.softmax(raw_scores / self.scale, dim=-1)
        attn = torch.matmul(attn_scores, value_proj)
        attn = attn.permute(0, 2, 1, 3).contiguous()
        new_shape = attn.size()[:-2] + (self.d_model,)
        attn = attn.view(*new_shape)
        output = (attn, )
        if get_attention_scores:
            output = output + (attn_scores, )
        return output


class TransformerEmbedding(nn.Module):
    def __init__(
            self,
            vocab_size,
            d_model,
            max_length,
            pad_token_id,
            token_type_vocab_size,
            pos_embedding_type='embedding',
            dropout=0.1,
            use_layer_norm=False,
            ln_eps=1e-12
        ):
        super(TransformerEmbedding, self).__init__()
        assert pos_embedding_type in ['embedding', 'timing']
        # TODO: implement timing signal
        self.token_embedding = nn.Embedding(vocab_size, d_model, pad_token_id)
        self.positional_embedding = nn.Embedding(max_length, d_model)
        self.token_type_embedding = nn.Embedding(token_type_vocab_size, d_model)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_ids", torch.arange(max_length).expand((1, -1)))

        self.layer_norm = nn.LayerNorm(d_model, eps=ln_eps)

        self.use_layer_norm = use_layer_norm

    def forward(
            self,
            input_ids,
            token_type_ids=None
        ):
        token_emb = self.token_embedding(input_ids)
        input_shape = token_emb.shape
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.pos_ids.device)
        token_emb = token_emb + self.token_type_embedding(token_type_ids)
        
        pos_ids = self.pos_ids[:, :input_shape[1]]

        token_emb = token_emb + self.positional_embedding(pos_ids)

        if self.use_layer_norm:
            token_emb = self.layer_norm(token_emb)
        
        token_emb = self.dropout(token_emb)
        return token_emb


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            num_layers,
            encoder_class,
            **kwargs
        ):
        super(TransformerEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList([
            encoder_class(**kwargs) for _ in range(num_layers)
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

