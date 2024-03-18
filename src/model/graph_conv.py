import torch
import numpy as np
import torch.nn as nn
import torch_scatter as tsc
import torch.nn.functional as F

from model.aggregator import *
from utils.dataset import *
from utils.parser import *

class GraphAggregateLayers(nn.Module):
    def __init__(self, args, dataset: Dataset):
        super().__init__()

        self._n_users = dataset.n_users
        self._n_items = dataset.n_items
        self._n_entities = dataset.n_entities
        self._n_relations = dataset.n_relations
        
        self._init_args(args)
        
        self._init_weights(args)
        
        self._kg_aggregator = Aggregator(args.emb_size, agg_type=args.kg_agg_type)
        self._cg_aggregator = Aggregator(args.emb_size, agg_type=args.cg_agg_type)

        self._drop_out = nn.Dropout(p=args.mess_dropout_rate)

        self._final_rep_generator = FinalRepGenerator(args.rep_type)
    
    def _init_args(self, args):
        self._device = args.device
        self._emb_size = args.emb_size
        
        self._n_hops_kg = args.n_hops_kg
        self._n_hops_cg = args.n_hops_cg
        
        self._node_dr_kg = args.node_dr_kg
        self._node_dr_cg = args.node_dr_cg
        
        self._n_heads = args.n_heads
        self._d_k = self._emb_size // self._n_heads
        
        self._rep_type = args.rep_type
    
    def _init_weights(self, args):
        self._relation_embs = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self._n_relations, 
                                                                               self._emb_size)))
        self._W_Q = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self._emb_size, 
                                                                     self._emb_size)))
        self._W_K = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self._emb_size, 
                                                                     self._emb_size)))
    
    def compute_all_item_attn_scores(self, 
                                     item_triplets,
                                     entity_embs):
        item_ids, relation_ids, entity_ids = item_triplets
        
        relation_entity_embs = self._relation_embs[relation_ids] * entity_embs[entity_ids]
        
        queries = torch.matmul(relation_entity_embs, self._W_Q).view(-1, self._n_heads, self._d_k)
        keys = torch.matmul(entity_embs[item_ids], self._W_K).view(-1, self._n_heads, self._d_k)

        raw_scores = torch.sum(queries * keys, dim=-1) / math.sqrt(self._d_k) # [n_item_triplets, n_heads]
        attn_scores = tsc.scatter_softmax(src=raw_scores, 
                                          index=item_ids, 
                                          dim=0).mean(dim=-1, keepdim=True) # [n_item_triplets, 1]
        
        degree_counter = tsc.scatter_sum(src=torch.ones_like(item_ids), 
                                         index=item_ids, 
                                         dim=0, 
                                         dim_size=self._n_items)
        scale_factor = torch.index_select(degree_counter, 
                                          dim=0, 
                                          index=item_ids)
        scaled_attn_scores = scale_factor.unsqueeze(dim=1) * attn_scores # [n_item_triplets, 1]

        return raw_scores, attn_scores, scaled_attn_scores

    def _aggregate_cg(self,
                      interaction_matrix,
                      embeddings):
        agg_embs = torch.sparse.mm(interaction_matrix, embeddings)
        return agg_embs
    
    def forward_cg(self, 
                   user_adj_mat, 
                   item_adj_mat, 
                   user_embs, 
                   item_embs, 
                   mess_dropout=False
                   ):
        
        cur_layer_user_embs = user_embs
        user_final_embs_cg = user_embs

        cur_layer_item_embs = item_embs
        item_final_embs_cg = item_embs

        for h in range(1, self._n_hops_cg+1):
            user_agg_embs = self._aggregate_cg(user_adj_mat, 
                                               cur_layer_item_embs)
            item_agg_embs = self._aggregate_cg(item_adj_mat, 
                                               cur_layer_user_embs)
            if mess_dropout:
                user_agg_embs = self._drop_out(user_agg_embs)
                item_agg_embs = self._drop_out(item_agg_embs)
            user_agg_embs = F.normalize(user_agg_embs)
            item_agg_embs = F.normalize(item_agg_embs)

            # update embeddings of the current layer
            cur_layer_user_embs = self._cg_aggregator(cur_layer_user_embs, user_agg_embs)
            cur_layer_item_embs = self._cg_aggregator(cur_layer_item_embs, item_agg_embs)

            user_final_embs_cg = self._final_rep_generator(user_final_embs_cg, cur_layer_user_embs, h)
            item_final_embs_cg = self._final_rep_generator(item_final_embs_cg, cur_layer_item_embs, h)
        
        return user_final_embs_cg, item_final_embs_cg
    
    def _is_at_high_hop(self, h):
        return h + 1 < self._n_hops_kg + 1
    
    def _holistic_aggregate(self,
                            kg_triplets,
                            entity_embs,
                            inter_mat,
                            user_embs,
                            h):
        # for normal entity nodes
        head_ids, relation_ids, tail_ids = kg_triplets
        entity_agg_embs = tsc.scatter_mean(src=self._relation_embs[relation_ids] * entity_embs[tail_ids],
                                           index=head_ids,
                                           dim=0,
                                           dim_size=self._n_entities) # [n_entities, emb_size]
        # for user representations
        user_agg_embs = self._aggregate_cg(inter_mat,
                                           entity_embs[:self._n_items])
        
        return entity_agg_embs, user_agg_embs
    
    def _agg_attn_based_embs(self, attn_scores, item_triplets, entity_embs, inter_mat, mess_dropout=False):
        item_ids, i_relation_ids, attribute_ids = item_triplets
        relation_attribute_embs = self._relation_embs[i_relation_ids] * entity_embs[attribute_ids]

        
        item_agg_embs = tsc.scatter_sum(src=attn_scores * relation_attribute_embs,
                                        index=item_ids,
                                        dim=0,
                                        dim_size=self._n_items) # [n_items, emb_size]
        if mess_dropout:
            item_agg_embs = self._drop_out(item_agg_embs)
        item_agg_embs = F.normalize(item_agg_embs)
        item_attn_final_embs = self._final_rep_generator(entity_embs[:self._n_items], item_agg_embs, 1)
        
        
        # for preference embs
           
        scaled_attn_scores = attn_scores # [n_item_triplets, 1]
        
        attribute_agg_embs = tsc.scatter_sum(src=scaled_attn_scores * relation_attribute_embs,
                                             index=item_ids,
                                             dim=0,
                                             dim_size=self._n_items)
        item_norm_scaled_attn_scores = tsc.scatter_sum(src=scaled_attn_scores,
                                                       index=item_ids,
                                                       dim=0,
                                                       dim_size=self._n_items) # [n_items, 1]
        preference_embs = torch.sparse.mm(inter_mat, 
                                          attribute_agg_embs) / (torch.sparse.mm(inter_mat, 
                                                                                 item_norm_scaled_attn_scores) + 1e-10)
        if mess_dropout:
            preference_embs = self._drop_out(preference_embs)
        preference_embs = F.normalize(preference_embs)

        return item_attn_final_embs, preference_embs
        
    def forward_holistic(self, 
                         kg_triplets,
                         item_triplets,
                         entity_embs, 
                         raw_scores,
                         inter_mat,
                         user_embs,
                         mess_dropout=False
                         ):
        cur_layer_entity_embs = entity_embs
        entity_final_embs = entity_embs

        attn_scores = tsc.scatter_softmax(src=raw_scores, index=item_triplets[0], dim=0).mean(dim=-1, keepdim=True)
          
        item_attn_final_embs, preference_embs = self._agg_attn_based_embs(attn_scores, item_triplets, entity_embs, inter_mat, mess_dropout)

        # for first-layer user agg
        cur_layer_user_embs, user_final_embs_kg = user_embs, user_embs
        
        for h in range(1, self._n_hops_kg+1):
            entity_agg_embs, user_agg_embs = self._holistic_aggregate(kg_triplets,
                                                                      cur_layer_entity_embs,
                                                                      inter_mat,
                                                                      cur_layer_user_embs,
                                                                      h)

            if mess_dropout:
                entity_agg_embs = self._drop_out(entity_agg_embs)   
                user_agg_embs = self._drop_out(user_agg_embs)
            entity_agg_embs = F.normalize(entity_agg_embs)
            user_agg_embs = F.normalize(user_agg_embs)

            cur_layer_entity_embs = self._kg_aggregator(cur_layer_entity_embs, entity_agg_embs)
            entity_final_embs = self._final_rep_generator(entity_final_embs, cur_layer_entity_embs, h)

            cur_layer_user_embs = self._kg_aggregator(cur_layer_user_embs, user_agg_embs)
            user_final_embs_kg = self._final_rep_generator(user_final_embs_kg, cur_layer_user_embs, h)

        return entity_final_embs, user_final_embs_kg, item_attn_final_embs, preference_embs