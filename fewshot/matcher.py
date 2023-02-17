import torch
import torch.nn as nn
from typing import Optional
from modules import EntityEncoder, RelationRepresentation, SoftSelectPrototype


class Matcher(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_symbols,
        use_pretrain=True,
        embed=None,
        dropout_layers=0.1,
        dropout_input=0.3,
        dropout_neighbors=0.0,
        finetune=False,
        num_transformer_layers=6,
        num_transformer_heads=4,
        device=torch.device("cpu"),
    ):
        super(Matcher, self).__init__()
        self.device = device
        self.EntityEncoder = EntityEncoder(
            embed_dim,
            num_symbols,
            use_pretrain=use_pretrain,
            embed=embed,
            dropout_input=dropout_input,
            dropout_neighbors=dropout_neighbors,
            finetune=finetune,
            device=device,
        )
        self.RelationRepresentation = RelationRepresentation(
            emb_dim=embed_dim,
            num_transformer_layers=num_transformer_layers,
            num_transformer_heads=num_transformer_heads,
            dropout_rate=dropout_layers,
        )
        self.Prototype = SoftSelectPrototype(embed_dim * num_transformer_heads)

    
    def forward(
        self,
        support,
        query,
        false=None,
        isEval=False,
        support_meta=None,
        query_meta=None,
        false_meta=None,
        task_emb: Optional[torch.Tensor] = None,
    ):
        """
        :param support:
        :param query:
        :param false:
        :param isEval:
        :param support_meta:
        :param query_meta:
        :param false_meta:
        :return:
        """
        if task_emb is not None:
            task_emb = task_emb.to(self.device)
        if not isEval:
            support_r = self.EntityEncoder(support, support_meta, task_emb=task_emb)
            query_r = self.EntityEncoder(query, query_meta, task_emb=task_emb)
            false_r = self.EntityEncoder(false, false_meta, task_emb=task_emb)

            support_r = self.RelationRepresentation(support_r[0], support_r[1])
            query_r = self.RelationRepresentation(query_r[0], query_r[1])
            false_r = self.RelationRepresentation(false_r[0], false_r[1])

            center_q = self.Prototype(support_r, query_r)
            center_f = self.Prototype(support_r, false_r)
            positive_score = torch.sum(query_r * center_q, dim=1)
            negative_score = torch.sum(false_r * center_f, dim=1)
        else:

            support_r = self.EntityEncoder(support, support_meta)
            query_r = self.EntityEncoder(query, query_meta)

            support_r = self.RelationRepresentation(support_r[0], support_r[1])
            query_r = self.RelationRepresentation(query_r[0], query_r[1])

            center_q = self.Prototype(support_r, query_r)
            positive_score = torch.sum(query_r * center_q, dim=1)
            negative_score = None
        return positive_score, negative_score
