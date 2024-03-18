import heapq
import multiprocessing
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from utils.dataset import *

# ============================================================================
def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)

def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num

def ndcg_at_k(r, k, ground_truth, method=1):
    def dcg_at_k(r, k, method=1):
        r = np.asfarray(r)[:k]
        if r.size:
            if method == 0:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            elif method == 1:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            else:
                raise ValueError('method must be 0 or 1.')
        return 0.

    GT = set(ground_truth)
    if len(GT) > k :
        sent_list = [1.0] * k
    else:
        sent_list = [1.0]*len(GT) + [0.0]*(k-len(GT))
    dcg_max = dcg_at_k(sent_list, k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def F1_at_k(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

def AUC(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res

def F1(ground_truth, y_prediction):
    try:
        res = f1_score(y_true=ground_truth, y_pred=y_prediction)
    except Exception:
        res = 0.
    return res

# ============================================================================
def test_one_user(user_scores):
    user = user_scores[0]
    ratings = user_scores[1]

    user_pos_record = user_record_gl[user]
    candidate_item_list = candidate_items_gl[user]

    item_score_map = {}
    for item in candidate_item_list:
        item_score_map[item] = ratings[item]
    
    item_scores_sorted = sorted(item_score_map.items(), key=lambda kv: kv[1], reverse=True)
    item_sorted = [x[0] for x in item_scores_sorted]
    scores_sorted = [x[1] for x in item_scores_sorted]

    r = []
    for i in item_sorted:
        if i in user_pos_record:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=scores_sorted)

    recall, ndcg = [], [] 

    for k in k_list:
        recall.append(recall_at_k(r, k, len(user_pos_record)))
        ndcg.append(ndcg_at_k(r, k, user_pos_record))


    return {'auc': auc, 
            'recall': np.array(recall), 
            'ndcg': np.array(ndcg)}
# ============================================================================

def eval_ctr_topk(args, 
                  model, 
                  dataset: Dataset, 
                  all_user_embs, 
                  all_item_embs):
    model.eval()
    
    global k_list, user_record_gl, candidate_items_gl
    k_list = eval(args.k_list)
    user_record_gl = dataset.test_user_record
    candidate_items_gl = dataset.user_candidate_items

    result = {'auc': 0.,
              'recall': np.zeros(len(k_list)),
              'ndcg': np.zeros(len(k_list)), 
            }
    
    cores = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(cores)

    user_list = list(dataset.test_user_record.keys())
    n_users = len(user_list)
    n_items = dataset.n_items
    
    count = 0
    u_start = 0
    while u_start < n_users:
        u_end = min(u_start+args.test_batch_size, n_users)
        user_batch = user_list[u_start:u_end]
        user_ids = torch.tensor(user_batch, dtype=torch.long).to(args.device)
        user_embs = all_user_embs[user_ids]
        
        user_batch_scores = np.zeros(shape=(user_ids.shape[0], n_items))
        
        i_count = 0
        i_start = 0
        while i_start < n_items:
            i_end = min(i_start+args.test_batch_size, n_items)
            item_ids = torch.tensor(np.array(range(i_start, i_end)), dtype=torch.long).view(i_end-i_start).to(args.device)
            item_embs = all_item_embs[item_ids]

            item_batch_pred_scores = model.rating_users_all_items(user_embs, item_embs)
            
            user_batch_scores[:, i_start:i_end] = item_batch_pred_scores
            
            i_count += item_batch_pred_scores.shape[1]

            i_start = i_end
        
        assert i_count == n_items

        user_id_batch_scores = zip(user_batch, user_batch_scores)
        batch_result = pool.map(test_one_user, user_id_batch_scores)
        count += len(batch_result)

        for re in batch_result:
            result['auc'] += re['auc'] / n_users
            result['recall'] += re['recall'] / n_users
            result['ndcg'] += re['ndcg'] / n_users
            
        
        u_start = u_end
    
    assert count == n_users
    
    pool.close()
    
    model.train()

    return result
