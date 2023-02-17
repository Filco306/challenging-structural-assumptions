import numpy as np
import logging
import torch
import torch.nn as nn
import math
import torch.nn.init as init
from typing import List, Optional


class AttentionSelectContext(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super(AttentionSelectContext, self).__init__()
        self.Bilinear = nn.Bilinear(dim, dim, 1, bias=False)
        self.Linear_tail = nn.Linear(dim, dim, bias=False)
        self.Linear_head = nn.Linear(dim, dim, bias=False)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def intra_attention(self, head, rel, tail, mask):
        """

        :param head: [b, dim]
        :param rel: [b, max, dim]
        :param tail:
        :param mask:
        :return:
        """
        head = head.unsqueeze(1).repeat(1, rel.size(1), 1)
        score = self.Bilinear(head, rel).squeeze(2)

        score = score.masked_fill_(mask, -np.inf)
        att = torch.softmax(score, dim=1).unsqueeze(dim=1)  # [b, 1, max]

        head = torch.bmm(att, tail).squeeze(1)
        return head

    
    def forward(
        self,
        left: List[torch.Tensor],
        right: List[torch.Tensor],
        mask_left=None,
        mask_right=None,
        task_emb: Optional[torch.Tensor] = None,
    ):
        """
        :param left: (head, rel, tail)
        :param right:
        :param mask_right:
        :param mask_left:
        :return:
        """
        head_left, rel_left, tail_left = left
        head_right, rel_right, tail_right = right

        # This should be exchanged!! It simply gives us more information, right?
        # weak_rel = head_right - head_left
        weak_rel = (
            torch.stack([task_emb] * head_left.shape[0])
            if task_emb is not None
            else (head_right - head_left)
        )
        # ipdb.sset_trace()
        left = self.intra_attention(weak_rel, rel_left, tail_left, mask_left)
        right = self.intra_attention(weak_rel, rel_right, tail_right, mask_right)

        left = torch.relu(self.Linear_tail(left) + self.Linear_head(head_left))
        right = torch.relu(self.Linear_tail(right) + self.Linear_head(head_right))

        left = self.dropout(left)
        right = self.dropout(right)

        left = self.layer_norm(left + head_left)
        right = self.layer_norm(right + head_right)
        return left, right


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention
    """

    def __init__(self, attn_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        :param attn_mask: [batch, time]
        :param scale:
        :param q: [batch, time, dim]
        :param k: [batch, time, dim]
        :param v: [batch, time, dim]
        :return:
        """
        attn = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attn = attn * scale
        if attn_mask:
            attn = attn.masked_fill_(attn_mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """ Implement without batch dim"""

    def __init__(self, model_dim, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads

        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        """
        To be efficient, multi- attention is cal-ed in a matrix totally
        :param attn_mask:
        :param query: [batch, time, per_dim * num_heads]
        :param key:
        :param value:
        :return: [b, t, d*h]
        """
        residual = query
        batch_size = key.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)

        scale = (key.size(-1) // self.num_heads) ** -0.5
        context, attn = self.dot_product_attention(query, key, value, scale, attn_mask)
        context = context.view(batch_size, -1, self.dim_per_head * self.num_heads)
        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        return output, attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()

        position_encoding = np.array(
            [
                [
                    pos / np.power(10000, 2.0 * (j // 2) / d_model)
                    for j in range(d_model)
                ]
                for pos in range(max_seq_len)
            ]
        )

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        pad_row = torch.zeros([1, d_model], dtype=torch.float)
        position_encoding = torch.tensor(position_encoding, dtype=torch.float)

        position_encoding = torch.cat((pad_row, position_encoding), dim=0)

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(
            position_encoding, requires_grad=False
        )

    def forward(self, batch_len, seq_len):
        """
        :param batch_len: scalar
        :param seq_len: scalar
        :return: [batch, time, dim]
        """
        input_pos = torch.tensor(
            [list(range(1, seq_len + 1)) for _ in range(batch_len)]
        )
        if torch.cuda.is_available():
            input_pos = input_pos.cuda()
        return self.position_encoding(input_pos)


class GELU(nn.Module):
    """
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
            )
        )


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.gelu = GELU()

    def forward(self, x):
        """

        :param x: [b, t, d*h]
        :return:
        """
        output = x.transpose(1, 2)  # [b, d*h, t]
        output = self.w2(self.gelu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        # ffn_dim
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        output = self.feed_forward(context)
        return output, attention


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        model_dim=100,
        ffn_dim=800,
        num_heads=4,
        dropout=0.1,
        num_layers=6,
        max_seq_len=3,
        with_pos=True,
    ):
        super(TransformerEncoder, self).__init__()
        self.with_pos = with_pos
        self.num_heads = num_heads

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(model_dim * num_heads, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.rel_embed = nn.Parameter(torch.rand(1, model_dim), requires_grad=True)

    def repeat_dim(self, emb):
        """
        :param emb: [batch, t, dim]
        :return:
        """
        return emb.repeat(1, 1, self.num_heads)

    def forward(self, left, right):
        """
        :param left: [batch, dim]
        :param right: [batch, dim]
        :return:
        """
        batch_size = left.size(0)
        rel_embed = self.rel_embed.expand_as(left)

        left = left.unsqueeze(1)
        right = right.unsqueeze(1)
        rel_embed = rel_embed.unsqueeze(1)  # [batch, 1, dim]

        seq = torch.cat((left, rel_embed, right), dim=1)
        pos = self.pos_embedding(batch_len=batch_size, seq_len=3)
        if self.with_pos:
            output = seq + pos
        else:
            output = seq
        output = self.repeat_dim(output)
        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output)
            attentions.append(attention)
        return output[:, 1, :]


class SoftSelectAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SoftSelectAttention, self).__init__()

    def forward(self, support, query):
        """
        :param support: [few, dim]
        :param query: [batch, dim]
        :return:
        """
        query_ = (
            query.unsqueeze(1)
            .expand(query.size(0), support.size(0), query.size(1))
            .contiguous()
        )  # [b, few, dim]
        support_ = support.unsqueeze(0).expand_as(query_).contiguous()  # [b, few, dim]

        scalar = support.size(1) ** -0.5  # dim ** -0.5
        score = torch.sum(query_ * support_, dim=2) * scalar
        att = torch.softmax(score, dim=1)

        center = torch.mm(att, support)
        return center


class SoftSelectPrototype(nn.Module):
    def __init__(self, r_dim):
        super(SoftSelectPrototype, self).__init__()
        self.Attention = SoftSelectAttention(hidden_size=r_dim)

    def forward(self, support, query):
        center = self.Attention(support, query)
        return center


class EntityEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_symbols,
        use_pretrain=True,
        embed=None,
        dropout_input=0.3,
        finetune=False,
        dropout_neighbors=0.0,
        device=torch.device("cpu"),
        using_transformer_embs: bool = False,
    ):
        super(EntityEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(
            num_symbols + 1, embed_dim, padding_idx=self.pad_idx
        )
        self.num_symbols = num_symbols

        self.gcn_w = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))
        self.dropout = nn.Dropout(dropout_input)
        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        self.pad_tensor = torch.tensor([self.pad_idx], requires_grad=False).to(device)

        if use_pretrain:
            logging.info("LOADING KB EMBEDDINGS")
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                logging.info("FIX KB EMBEDDING")
                self.symbol_emb.weight.requires_grad = False

        self.NeighborAggregator = AttentionSelectContext(
            dim=embed_dim, dropout=dropout_neighbors
        )

    def neighbor_encoder_mean(self, connections, num_neighbors):
        """
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        """
        num_neighbors = num_neighbors.unsqueeze(1)
        relations = connections[:, :, 0].squeeze(-1)
        entities = connections[:, :, 1].squeeze(-1)
        rel_embeds = self.dropout(self.symbol_emb(relations))
        ent_embeds = self.dropout(self.symbol_emb(entities))

        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)
        out = self.gcn_w(concat_embeds)

        out = torch.sum(out, dim=1)
        out = out / num_neighbors
        return out.tanh()

    def neighbor_encoder_soft_select(
        self, connections_left, connections_right, head_left, head_right, task_emb
    ):
        """
        :param connections_left: [b, max, 2]
        :param connections_right:
        :param head_left:
        :param head_right:
        :return:
        """
        relations_left = connections_left[:, :, 0].squeeze(-1)
        entities_left = connections_left[:, :, 1].squeeze(-1)
        rel_embeds_left = self.dropout(self.symbol_emb(relations_left))  # [b, max, dim]
        ent_embeds_left = self.dropout(self.symbol_emb(entities_left))

        pad_matrix_left = self.pad_tensor.expand_as(relations_left)
        mask_matrix_left = torch.eq(relations_left, pad_matrix_left).squeeze(
            -1
        )  # [b, max]

        relations_right = connections_right[:, :, 0].squeeze(-1)
        entities_right = connections_right[:, :, 1].squeeze(-1)
        rel_embeds_right = self.dropout(
            self.symbol_emb(relations_right)
        )  # (batch, 200, embed_dim)
        ent_embeds_right = self.dropout(
            self.symbol_emb(entities_right)
        )  # (batch, 200, embed_dim)

        pad_matrix_right = self.pad_tensor.expand_as(relations_right)
        mask_matrix_right = torch.eq(relations_right, pad_matrix_right).squeeze(
            -1
        )  # [b, max]

        left = [head_left, rel_embeds_left, ent_embeds_left]
        right = [head_right, rel_embeds_right, ent_embeds_right]
        output = self.NeighborAggregator(
            left, right, mask_matrix_left, mask_matrix_right, task_emb=task_emb
        )
        return output

    def forward(self, entity, entity_meta=None, task_emb=None):
        """
         query: (batch_size, 2)
         entity: (few, 2)
         return: (batch_size, )
         """
        if entity_meta is not None:
            entity = self.symbol_emb(entity)
            (
                entity_left_connections,
                entity_left_degrees,
                entity_right_connections,
                entity_right_degrees,
            ) = entity_meta
            entity_left, entity_right = torch.split(entity, 1, dim=1)
            entity_left = entity_left.squeeze(1)
            entity_right = entity_right.squeeze(1)
            entity_left, entity_right = self.neighbor_encoder_soft_select(
                entity_left_connections,
                entity_right_connections,
                entity_left,
                entity_right,
                task_emb=task_emb,
            )
        else:
            # no_meta
            entity = self.symbol_emb(entity)
            entity_left, entity_right = torch.split(entity, 1, dim=1)
            entity_left = entity_left.squeeze(1)
            entity_right = entity_right.squeeze(1)
        return entity_left, entity_right


class RelationRepresentation(nn.Module):
    def __init__(
        self, emb_dim, num_transformer_layers, num_transformer_heads, dropout_rate=0.1
    ):
        super(RelationRepresentation, self).__init__()
        self.RelationEncoder = TransformerEncoder(
            model_dim=emb_dim,
            ffn_dim=emb_dim * num_transformer_heads * 2,
            num_heads=num_transformer_heads,
            dropout=dropout_rate,
            num_layers=num_transformer_layers,
            max_seq_len=3,
            with_pos=True,
        )

    def forward(self, left, right):
        """
        forward
        :param left: [batch, dim]
        :param right: [batch, dim]
        :return: [batch, dim]
        """

        relation = self.RelationEncoder(left, right)
        return relation
