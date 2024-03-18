import torch
import torch.nn as nn
import torch.nn.functional as F

from model.graph_conv import *
from utils.dataset import *
from utils.parser import *


class KPRec(nn.Module):
    def __init__(self, args, dataset: Dataset):
        super().__init__()
        
        # init parameters
        self._n_items = dataset.n_items
        self._n_users = dataset.n_users
        self._device = args.device

        self._emb_size = args.emb_size
        self._l2_decay = args.l2_decay
        self._cg_decay = args.cg_decay
        self._cl_decay = args.cl_decay
        self._tau = args.tau

        self._node_dr_cg = args.node_dr_cg
        self._node_dr_kg = args.node_dr_kg
        self._node_dr_pf = args.node_dr_pf
        self._attn_dr = args.attn_dr
        
        self._kg_triplets = torch.tensor(dataset._triplet_array).long().to(self._device) # [3, n_triplets]
        self._item_triplets = self._filter_item_triplets(self._kg_triplets)

        self._user_adj_mat = self._convert_sp_mat_to_sp_tensor(dataset.user_adj_mat).to(self._device)
        self._item_adj_mat = self._convert_sp_mat_to_sp_tensor(dataset.item_adj_mat).to(self._device)
        self._inter_mat    = self._convert_sp_mat_to_sp_tensor(dataset.inter_mat).to(self._device) # user adj mat, norm by dgree

        # init embedding weights
        self._user_embs = nn.Parameter(nn.init.xavier_uniform_(torch.empty(dataset.n_users, 
                                                                           self._emb_size)))
        self._entity_embs = nn.Parameter(nn.init.xavier_uniform_(torch.empty(dataset.n_entities, 
                                                                             self._emb_size)))

        self._gnn = GraphAggregateLayers(args, dataset)
    
    def _filter_item_triplets(self, all_triplets):
        indices = torch.where(all_triplets[0] < self._n_items)[0]
        return all_triplets[:, indices]
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        
        return torch.sparse.FloatTensor(i, v, coo.shape)
     
    def _cg_dropout(self, user_adj_mat, item_adj_mat, rate=0.5):
        noise_shape = user_adj_mat._nnz()
        
        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(self._device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        
        indices_user = user_adj_mat._indices()
        values_user = user_adj_mat._values()
        indices_user = indices_user[:, dropout_mask]
        values_user = values_user[dropout_mask] / rate
        sampled_user_adj_mat = torch.sparse.FloatTensor(indices_user, 
                                                        values_user, 
                                                        user_adj_mat.shape).to(self._device) # [n_users, n_items]
        
        indices_item_t = item_adj_mat.t()._indices()
        values_item_t = item_adj_mat.t()._values()
        indices_item_t = indices_item_t[:, dropout_mask]
        values_item_t = values_item_t[dropout_mask] / rate
        sampled_item_adj_mat = torch.sparse.FloatTensor(indices_item_t, 
                                                        values_item_t, 
                                                        user_adj_mat.shape).t().to(self._device) #[n_items, n_users]
        
        return sampled_user_adj_mat, sampled_item_adj_mat
    
    def _sparse_dropout(self, sparse_mat, rate=0.5):
        noise_shape =  sparse_mat._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(self._device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)

        indices = sparse_mat._indices()
        values = sparse_mat._values()

        indices = indices[:, dropout_mask]
        values = values[dropout_mask] / rate 

        out = torch.sparse.FloatTensor(indices, 
                                       values,
                                       sparse_mat.shape).to(self._device)

        return out
    
    def _attn_based_dropout(self, scaled_attn_scores, item_triplets, raw_scores, rate=0.5, attn_ratio=0.2):
        n_edges = item_triplets.shape[1]
        
        noise = -torch.log(-torch.log(torch.rand_like(scaled_attn_scores)))
        scaled_attn_scores = scaled_attn_scores + noise
        last_k_values, last_k_indices = torch.topk(-scaled_attn_scores, dim=0, k=int(n_edges*rate*attn_ratio), sorted=False)
        last_k_indices = last_k_indices.squeeze().cpu().numpy()
        last_k_mask = np.zeros(n_edges, dtype=bool)
        last_k_mask[last_k_indices] = True

        random_indices = np.random.choice(n_edges, size=int(n_edges*rate*(1-attn_ratio)), replace=False)
        random_mask = np.zeros(n_edges, dtype=bool)
        random_mask[random_indices] = True 

        mask = last_k_mask | random_mask

        remain_triplets = item_triplets[:, ~mask]
        raw_scores = raw_scores[~mask]

        return remain_triplets, raw_scores
    
    def _kg_dropout(self, triplets, rate=0.5):
        n_edges = triplets.shape[1]
        chosen_indices = np.random.choice(n_edges, size=int(n_edges*rate), replace=False)
        return triplets[:, chosen_indices]

    def _aggregate_graphs(self, mess_dropout=False, node_dropout=False):
        kg_triplets = self._kg_triplets
        item_triplets = self._item_triplets
        inter_mat = self._inter_mat
        user_adj_mat = self._user_adj_mat
        item_adj_mat = self._item_adj_mat
        
        raw_scores, attn_scores, scaled_attn_scores = self._gnn.compute_all_item_attn_scores(item_triplets, self._entity_embs)

        if node_dropout:
            item_triplets, raw_scores = self._attn_based_dropout(scaled_attn_scores, item_triplets, raw_scores, self._node_dr_pf, self._attn_dr)

            kg_triplets = self._kg_dropout(self._kg_triplets, self._node_dr_kg)
            inter_mat = self._sparse_dropout(self._inter_mat, self._node_dr_cg)
            user_adj_mat, item_adj_mat = self._cg_dropout(self._user_adj_mat, self._item_adj_mat)
        
        entity_final_embs, user_final_embs_kg, \
            item_attn_final_embs, preference_embs = self._gnn.forward_holistic(kg_triplets,
                                                                               item_triplets,
                                                                               self._entity_embs,
                                                                               raw_scores,
                                                                               inter_mat,
                                                                               self._user_embs,
                                                                               mess_dropout)
        
        user_final_embs_cg, item_final_embs_cg = self._gnn.forward_cg(user_adj_mat,
                                                                      item_adj_mat,
                                                                      self._user_embs,
                                                                      self._entity_embs[:self._n_items],
                                                                      mess_dropout)
        
        
        return entity_final_embs, user_final_embs_kg, \
                user_final_embs_cg, item_final_embs_cg, item_attn_final_embs, preference_embs

    def forward(self, 
                user_ids, 
                item_ids, 
                neg_item_ids, 
                mess_dropout=False, 
                node_dropout=False):
        
        entity_final_embs, user_final_embs_kg, \
                user_final_embs_cg, item_final_embs_cg, item_attn_final_embs, preference_embs = self._aggregate_graphs(mess_dropout, node_dropout)
        
        user_ids = torch.tensor(user_ids, dtype=torch.long).to(self._device)
        item_ids = torch.tensor(item_ids, dtype=torch.long).to(self._device)
        neg_item_ids = torch.tensor(neg_item_ids, dtype=torch.long).to(self._device)

        loss = self._object_func(user_ids,
                                 item_ids,
                                 neg_item_ids,
                                 entity_final_embs,
                                 user_final_embs_kg,
                                 user_final_embs_cg,
                                 preference_embs,
                                 item_final_embs_cg,
                                 item_attn_final_embs)
        
        return loss
    
    def _constrast_loss(self, item_final_embs_cg, item_attn_final_embs, item_final_embs_kg,
                        user_final_embs_cg, user_final_embs_kg, preference_embs):
        def _sim(z1, z2):
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.exp((z1 * z2).sum(dim=1) / self._tau)
        
        item_pos_sim = _sim(item_attn_final_embs, item_final_embs_cg)
        item_neg_sim = _sim(item_attn_final_embs, item_final_embs_kg)
        
        item_contr_loss = -torch.log(item_pos_sim  / (item_pos_sim  + item_neg_sim)).sum()

        user_pos_sim = _sim(preference_embs, user_final_embs_cg)
        user_neg_sim = _sim(preference_embs, user_final_embs_kg) 

        user_contr_loss = -torch.log(user_pos_sim / (user_pos_sim + user_neg_sim)).sum()
        
        return item_contr_loss + user_contr_loss
    
    def _compute_scores(self, user_embs, item_embs):
        return torch.sum(user_embs * item_embs, dim=1)
    
    def _get_user_total_rep(self, user_embs_kg, user_embs_cg, user_preference_embs):        
        user_embs_total = torch.concat(((user_preference_embs + user_embs_cg) / 2, user_embs_kg), dim=1)
 
        return user_embs_total

    def _get_item_total_rep(self, item_embs_kg, item_embs_cg, item_attn_embs):
        item_embs_total = torch.concat(((item_embs_cg + item_attn_embs) / 2, item_embs_kg), dim=1)
        
        return item_embs_total
    
    def _object_func(self,
                     user_ids,
                     item_ids,
                     neg_item_ids,
                     entity_final_embs,
                     user_final_embs_kg,
                     user_final_embs_cg,
                     preference_embs,
                     item_final_embs_cg,
                     item_attn_final_embs):
        
        cur_batch_size = len(user_ids)

        user_embs_kg = user_final_embs_kg[user_ids]

        item_embs_attn = item_attn_final_embs[item_ids]
        neg_item_embs_attn = item_attn_final_embs[neg_item_ids]
        user_preference_embs = preference_embs[user_ids]   

        item_embs_kg = entity_final_embs[item_ids]
        neg_item_embs_kg = entity_final_embs[neg_item_ids]

        
        user_embs_cg = user_final_embs_cg[user_ids]
        item_embs_cg = item_final_embs_cg[item_ids]
        neg_item_embs_cg = item_final_embs_cg[neg_item_ids]
        
        # bpr loss
        user_embs_total = self._get_user_total_rep(user_embs_kg, user_embs_cg, user_preference_embs)
        item_embs_total = self._get_item_total_rep(item_embs_kg, item_embs_cg, item_embs_attn)
        neg_item_embs_total = self._get_item_total_rep(neg_item_embs_kg, neg_item_embs_cg, neg_item_embs_attn)

        pos_pred_scores = self._compute_scores(user_embs_total, item_embs_total)
        neg_pred_scores = self._compute_scores(user_embs_total, neg_item_embs_total)
        
        bpr_loss = -1 * torch.mean(nn.LogSigmoid()(pos_pred_scores - neg_pred_scores))

        # cg bpr loss
        pos_scores_cg = self._compute_scores(user_embs_cg, item_embs_cg)
        neg_scores_cg = self._compute_scores(user_embs_cg, neg_item_embs_cg)
        cg_bpr_loss = self._cg_decay * -1 * torch.mean(nn.LogSigmoid()(pos_scores_cg - neg_scores_cg))
        
        # contrastive loss
        
        cl_loss = self._cl_decay * self._constrast_loss(item_final_embs_cg,
                                                        item_attn_final_embs, 
                                                        entity_final_embs[:self._n_items],
                                                        user_final_embs_cg, 
                                                        user_final_embs_kg,
                                                        preference_embs)
        
        # regularization
        regularizer = (torch.norm(user_embs_kg)**2 +  torch.norm(user_embs_cg)**2 +
                       torch.norm(item_embs_kg)**2 +  torch.norm(item_embs_cg)**2 +
                       torch.norm(neg_item_embs_kg)**2 + torch.norm(neg_item_embs_cg)**2 +
                       torch.norm(user_preference_embs) ** 2 +
                       torch.norm(item_embs_attn)**2 + torch.norm(neg_item_embs_attn)**2
                       ) / 2
        emb_loss = self._l2_decay * (regularizer / cur_batch_size)

        loss = bpr_loss + cg_bpr_loss + cl_loss + emb_loss

        return loss, bpr_loss, cg_bpr_loss, cl_loss, emb_loss
    

    def generate_combined_embs(self):
        self.eval()
        entity_final_embs, user_final_embs_kg, \
            user_final_embs_cg, item_final_embs_cg, item_attn_final_embs, preference_embs = self._aggregate_graphs()
        
        user_embs_total = self._get_user_total_rep(user_final_embs_kg, user_final_embs_cg, preference_embs)
        item_embs_total = self._get_item_total_rep(entity_final_embs[:self._n_items], item_final_embs_cg, item_attn_final_embs)
        
        self.train()
        
        return user_embs_total, item_embs_total
    
    def rating_users_all_items(self, user_preference_embs, item_ck_embs):
        pred_scores_mat = torch.matmul(user_preference_embs, item_ck_embs.t())
        return pred_scores_mat.detach().cpu().numpy()