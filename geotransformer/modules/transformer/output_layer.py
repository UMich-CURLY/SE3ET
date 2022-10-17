import torch.nn as nn

from geotransformer.modules.layers import build_act_layer, build_dropout_layer
from einops import rearrange


class AttentionOutput(nn.Module):
    def __init__(self, d_model, dropout=None, activation_fn='ReLU'):
        super(AttentionOutput, self).__init__()
        self.expand = nn.Linear(d_model, d_model * 2)
        self.activation = build_act_layer(activation_fn)
        self.squeeze = nn.Linear(d_model * 2, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_states):
        hidden_states = self.expand(input_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.squeeze(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(input_states + hidden_states)
        return output_states

class RotCompressOutput(nn.Module):
    def __init__(self, d_model, dropout=None, activation_fn='ReLU', na=12, dual_align=False):
        super(RotCompressOutput, self).__init__()
        self.na = na
        self.dual_align = dual_align
        if dual_align:
            self.expand = nn.Linear(d_model * na * 2, d_model * 2)
        else:
            self.expand = nn.Linear(d_model * na, d_model * 2)
        self.activation = build_act_layer(activation_fn)
        self.squeeze = nn.Linear(d_model * 2, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_states):
        """input_states: banc"""
        input_states_max, _ = input_states.max(1)  # bnc
        input_states = rearrange(input_states, 'b a n c -> b n (a c)')
        hidden_states = self.expand(input_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.squeeze(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(input_states_max + hidden_states)
        return output_states
