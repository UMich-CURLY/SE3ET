import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class RotationAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(RotationAttentionLayer, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)

    def forward(
        self, ref_feats_m, src_feats_m, ref_node_corr_indices, src_node_corr_indices):

        print('input ref_feats_m', ref_feats_m.shape)
        
        ref_feats_m = rearrange(self.proj_q(ref_feats_m), 'b a n (h c) -> b a h n c', h=self.num_heads)
        src_feats_m = rearrange(self.proj_k(src_feats_m), 'b a m (h c) -> b a h m c', h=self.num_heads)
        print('rearranged ref_feats_m', ref_feats_m.shape)

        ref_matching = ref_feats_m[:, :, :, ref_node_corr_indices, :] # bahnc -> bahn'c
        src_matching = src_feats_m[:, :, :, src_node_corr_indices, :] # behmc -> behn'c, find the best matching point
        print('ref_matching', ref_matching.shape)

        # normalize over nc
        temp_ref_matching = F.normalize(rearrange(ref_matching, 'b a h n c -> b a h (n c)'), dim=-1)
        print('temp_ref_matching', temp_ref_matching.shape, 'n', ref_node_corr_indices.shape[0])
        ref_matching = rearrange(temp_ref_matching, 'b a h (n c) -> b a h n c', c=ref_matching.shape[-1])
        temp_src_matching = F.normalize(rearrange(src_matching, 'b a h m c -> b a h (m c)'), dim=-1)
        print('temp_src_matching', temp_src_matching.shape, 'm', src_node_corr_indices.shape[0])
        src_matching = rearrange(temp_src_matching, 'b a h (m c) -> b a h m c', c=src_matching.shape[-1])

        # calculate attention matrix
        attention_scores_ae_rot_sup = torch.einsum('bahnc,behnc->baeh', ref_matching, src_matching)
        attention_scores_ae_rot_sup = attention_scores_ae_rot_sup.mean(3) # bae
        attention_scores_ae_rot_sup = (attention_scores_ae_rot_sup + 1) / 2

        return attention_scores_ae_rot_sup