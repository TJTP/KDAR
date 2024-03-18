import torch
import torch.nn as nn
import torch.nn.functional as F

class Aggregator(nn.Module):
    def __init__(self, emb_size, agg_type='neigh'):
        super().__init__()
        self._agg_type = agg_type
        self._initializer = nn.init.xavier_uniform_
        
        if self._agg_type == 'gcn':
            self._gcn_W = nn.Parameter(self._initializer(torch.empty(emb_size, emb_size)))
            self._gcn_b = nn.Parameter(self._initializer(torch.empty(1, emb_size)))    
        elif self._agg_type == 'bi':
            self._bi_W_add = nn.Parameter(self._initializer(torch.empty(emb_size, emb_size)))
            self._bi_b_add = nn.Parameter(self._initializer(torch.empty(1,emb_size)))
            self._bi_W_concat = nn.Parameter(self._initializer(torch.empty(emb_size * 2, emb_size)))
            self._bi_b_concat = nn.Parameter(self._initializer(torch.empty(1, emb_size)))
        elif self._agg_type == 'concat':
            self._concat_W = nn.Parameter(self._initializer(torch.empty(2 * emb_size, emb_size)))
            self._concat_b = nn.Parameter(self._initializer(torch.empty(1, emb_size)))    
        
    def __call__(self, ego_embs, agg_embs):
        if self._agg_type == 'gcn':
            new_ego_embs = F.elu(torch.matmul(ego_embs + agg_embs, self._gcn_W) + self._gcn_b)
        elif self._agg_type == 'sum':
            new_ego_embs = torch.add(ego_embs, agg_embs)
        elif self._agg_type == 'neigh':
            new_ego_embs = agg_embs
        elif self._agg_type == 'avg':
            new_ego_embs = torch.add(ego_embs, agg_embs) / 2
        elif self._agg_type == 'bi':
            new_ego_embs = F.leaky_relu(torch.matmul(ego_embs + agg_embs,self._bi_W_add) + self._bi_b_add) \
                + F.leaky_relu(torch.matmul(torch.concat([ego_embs, agg_embs], dim=-1), self._bi_W_concat) + self._bi_b_concat)
        elif self._agg_type == 'concat':
            new_ego_embs = F.elu(torch.matmul(torch.concat([ego_embs, agg_embs], dim=-1), self._concat_W) + self._concat_b)

        return new_ego_embs

class FinalRepGenerator():
    def __init__(self, rep_type='sumup'):
        super().__init__()
        self._rep_type = rep_type
    
    def __call__(self, pre_final_embs, cur_layer_embs, h):
        if self._rep_type == 'last':
            final_embs = cur_layer_embs
        elif self._rep_type == 'sumup':
            final_embs = torch.add(pre_final_embs, cur_layer_embs)
        elif self._rep_type == 'wsumup':
            final_embs = torch.add(pre_final_embs, (1 / h) * cur_layer_embs)
        elif self._rep_type == 'concatall':
            final_embs = torch.concat([pre_final_embs, cur_layer_embs], dim=-1)
        
        return final_embs