r"""Transformer with Relative Positional Embeddings.

Relative positional embedding is further projected in each multi-head attention layer.

The shape of input tensor should be (B, N, C). Implemented with `nn.Linear` and `nn.LayerNorm` (with affine).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from IPython import embed

from geotransformer.modules.layers import build_dropout_layer
from geotransformer.modules.transformer.output_layer import AttentionOutput


class RPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, equivariant=False, d_equiv_embed=0):
        super(RPEMultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads
        self.equivariant = equivariant

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(self.d_model, self.d_model)
        self.d_equiv_embed = d_equiv_embed
        if self.equivariant and self.d_equiv_embed > 0:
            self.proj_eq = nn.Linear(self.d_equiv_embed, self.d_model)

        self.dropout = build_dropout_layer(dropout)

    def forward(self, input_q, input_k, input_v, embed_qk, key_weights=None, key_masks=None, attention_factors=None, embed_eq=None):
        r"""Scaled Dot-Product Attention with Pre-computed Relative Positional Embedding (forward)

        Args:
            input_q: torch.Tensor (B, [A], N, C)
            input_k: torch.Tensor (B, [A], M, C)
            input_v: torch.Tensor (B, [A], M, C)
            embed_qk: torch.Tensor (B, N, M, C), relative positional embedding
            key_weights: torch.Tensor (B, M), soft masks for the keys
            key_masks: torch.Tensor (B, M), True if ignored, False if preserved
            attention_factors: torch.Tensor (B, N, M)
            embed_eq: torch.Tensor (B, A, N, M, C_eq)

        Returns:
            hidden_states: torch.Tensor (B, [A], N, C)
            attention_scores: torch.Tensor (B, [A], H, N, M)
        """
        if self.equivariant:
            q = rearrange(self.proj_q(input_q), 'b a n (h c) -> b a h n c', h=self.num_heads)
            k = rearrange(self.proj_k(input_k), 'b a m (h c) -> b a h m c', h=self.num_heads)
            v = rearrange(self.proj_v(input_v), 'b a m (h c) -> b a h m c', h=self.num_heads)
            p = rearrange(self.proj_p(embed_qk), 'b n m (h c) -> b h n m c', h=self.num_heads)

            # print('self attention input q (mean, std):', torch.mean(input_q), torch.std(input_q))
            # print('self attention q (mean, std):', torch.mean(q), torch.std(q))
            # print('self attention input k (mean, std):', torch.mean(input_k), torch.std(input_k))
            # print('self attention k (mean, std):', torch.mean(k), torch.std(k))
            # print('self attention input v (mean, std):', torch.mean(input_v), torch.std(input_v))
            # print('self attention v (mean, std):', torch.mean(v), torch.std(v))
            # print('self attention input p (mean, std):', torch.mean(embed_qk), torch.std(embed_qk))
            # print('self attention p (mean, std):', torch.mean(p), torch.std(p))
            
            attention_scores_p = torch.einsum('bahnc,bhnmc->bahnm', q, p)
            attention_scores_e = torch.einsum('bahnc,bahmc->bahnm', q, k)
            # print('self attention_scores_p (mean, std):', torch.mean(attention_scores_p), torch.std(attention_scores_p))
            # print('self attention_scores_e (mean, std):', torch.mean(attention_scores_e), torch.std(attention_scores_e))

            if self.d_equiv_embed > 0:
                assert embed_eq is not None, 'Equivariant embedding required here.'
                eq = rearrange(self.proj_eq(embed_eq), 'b a n m (h c) -> b a h n m c', h=self.num_heads)
                attention_scores_eq = torch.einsum('bahnc,bahnmc->bahnm', q, eq)
                # print('self attention input embed_eq (mean, std):', torch.mean(embed_eq), torch.std(embed_eq))
                # print('self attention eq (mean, std):', torch.mean(eq), torch.std(eq))
                # print('self attention_scores_eq (mean, std):', torch.mean(attention_scores_eq), torch.std(attention_scores_eq))
            

        else:
            q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
            k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
            v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)
            p = rearrange(self.proj_p(embed_qk), 'b n m (h c) -> b h n m c', h=self.num_heads)

            attention_scores_p = torch.einsum('bhnc,bhnmc->bhnm', q, p)
            attention_scores_e = torch.einsum('bhnc,bhmc->bhnm', q, k)

        if self.equivariant and self.d_equiv_embed > 0:
            attention_scores = (attention_scores_e + attention_scores_p + attention_scores_eq) / self.d_model_per_head ** 0.5
            # print('self attention_scores (mean, std):', torch.mean(attention_scores), torch.std(attention_scores))
            # print('self.d_model_per_head', self.d_model_per_head)
        else:
            attention_scores = (attention_scores_e + attention_scores_p) / self.d_model_per_head ** 0.5
            # print('self attention_scores (mean, std):', torch.mean(attention_scores), torch.std(attention_scores))

        if attention_factors is not None:
            if self.equivariant:
                attention_scores = attention_factors.unsqueeze(1).unsqueeze(1) * attention_scores
            else:
                attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_weights is not None:
            if self.equivariant:
                attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            else:
                attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            if self.equivariant:
                attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1).unsqueeze(-1), float('-inf'))
            else:
                attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        # print('self attention_scores after softmax mean', torch.mean(attention_scores))
        # print('self attention_scores after softmax std', torch.std(attention_scores))

        hidden_states = torch.matmul(attention_scores, v)

        if self.equivariant:
            hidden_states = rearrange(hidden_states, 'b a h n c -> b a n (h c)')
        else:
            hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

        ### if equivariant: b a n (h c), bhanm; else:  b n (h c), bhnm
        return hidden_states, attention_scores


class RPEAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, equivariant=False, d_equiv_embed=0):
        super(RPEAttentionLayer, self).__init__()
        self.attention = RPEMultiHeadAttention(d_model, num_heads, dropout=dropout, equivariant=equivariant, d_equiv_embed=d_equiv_embed)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
        equiv_states=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_states,
            position_states,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
            embed_eq=equiv_states,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class RPETransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU', equivariant=False, d_equiv_embed=0):
        super(RPETransformerLayer, self).__init__()
        self.attention = RPEAttentionLayer(d_model, num_heads, dropout=dropout, equivariant=equivariant, d_equiv_embed=d_equiv_embed)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
        equiv_states=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            position_states,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
            equiv_states=equiv_states,
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores
