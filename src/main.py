import os
import math
import random
import torch
import logging
import numpy as np
from time import time
from model.rec_model import KPRec
from utils.parser import parse_args
from utils.dataset import *
from utils.eval_metrics import eval_ctr_topk
from utils.helper import count_params, show_topk_info, early_stopping

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# set logger
logger = logging.getLogger('logger')
logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s", 
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
logger.setLevel(logging.INFO)

# ============================================================================================================
def negative_sampling(user_ids, user_candidate_items):
    neg_items = []
    for user_id in user_ids:
        neg_item = np.random.randint(low=0, high=len(user_candidate_items[user_id]), size=1)[0]
        neg_items.append(user_candidate_items[user_id][neg_item])
    return np.array(neg_items)

# ================================================================================================================================

if __name__ == '__main__':
    logger.info("================== All Args ====================")
    args = parse_args()
    logger.info(args)
    logger.info("=================================================")

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Seed: %d"%(args.seed))

    dataset = Dataset(args, logger)
    
    logger.info("================== Statistical features ====================")
    logger.info('# users:  %d'%(dataset.n_users))
    logger.info('# item:  %d'%(dataset.n_items))
    logger.info('# interactions: %d'%(dataset.train_data.shape[0]+dataset.test_data.shape[0]))
    logger.info('\t# train interactions: %d'%(dataset.train_data.shape[0]))
    logger.info('\t# test interactions:  %d'%(dataset.test_data.shape[0]))
    logger.info('# entities:  %d'%(dataset.n_entities))
    logger.info('# relations:  %d'%(dataset.n_relations))
    logger.info('# triplets:  %d'%(dataset.n_kg_triplets))

    
    logger.info("================== Init model ====================")
    model = KPRec(args, dataset)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logger.info("================== Training model ====================")
    
    
    tot_training_time = 0.
    tot_eval_time, eval_cnt = 0., 0

    best_recall_K_20 = -1.
    best_epoch = 0
    auc_rec, ndcg_rec = 0., 0.
    
    cur_best_pre_0 = 0.
    stopping_step = 0
    should_stop = False
    start_epoch = 0

    for epoch in range(start_epoch, args.n_epochs):
        
        t_s_train = time()

        # shuffle train data
        index = np.arange(dataset.train_data.shape[0])
        np.random.shuffle(index)
        dataset.train_data = dataset.train_data[index]
        
        loss_record = {'loss': [], 
                       'bpr': [], 
                       'cg_bpr': [],
                       'cl': [],
                       'emb': [], 
                       }
        
        start = 0
        while start < dataset.train_data.shape[0]:
            end = min(start+args.batch_size, dataset.train_data.shape[0])
            
            batch_train_data = dataset.train_data[start:end]
            user_ids, item_ids = batch_train_data[:, 0], batch_train_data[:, 1]

            neg_item_ids = negative_sampling(user_ids, dataset.user_candidate_items)
            
            loss, bpr_loss, cg_bpr_loss, cl_loss, emb_loss = model.forward(user_ids, 
                                                                           item_ids, 
                                                                           neg_item_ids, 
                                                                           mess_dropout=args.mess_dropout, 
                                                                           node_dropout=args.node_dropout)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_record['loss'].append(loss.item())
            loss_record['bpr'].append(bpr_loss.item())
            loss_record['cg_bpr'].append(cg_bpr_loss.item())
            loss_record['cl'].append(cl_loss.item())
            loss_record['emb'].append(emb_loss.item())
            
            start = end

        tot_loss = np.sum(loss_record['loss'])
        avg_loss = np.mean(loss_record['loss'])
        tot_bpr_loss = np.sum(loss_record['bpr'])
        tot_cg_bpr_loss = np.sum(loss_record['cg_bpr'])
        tot_cl_loss = np.sum(loss_record['cl'])
        tot_emb_loss = np.sum(loss_record['emb'])
        

        t_e_train = time()
        t_cur_epoch = t_e_train - t_s_train
        tot_training_time += t_cur_epoch

        logger.info('Epoch: <%.2d>, Training time: %.2f s, Loss: %.4f (avg: %.4f) = bpr: %.3f + cgbpr: %.3f +  cl: %.3f + emb: %.3f'
                        %(epoch+1, t_cur_epoch, tot_loss, avg_loss, tot_bpr_loss, tot_cg_bpr_loss, tot_cl_loss, tot_emb_loss))
        
        if (epoch + 1) % args.show_steps == 0 or (epoch + 1) == 1:
            logger.info('********************************************************************************')
            t_s_eval = time()
            with torch.no_grad():
                all_user_preference_embs, all_item_ck_embs = model.generate_combined_embs()
                
                test_result = eval_ctr_topk(args, model, dataset, all_user_preference_embs, all_item_ck_embs)

                logger.info('Epoch %.2d    AUC: %.4f'%(epoch+1, test_result['auc']))
                show_topk_info(args, logger, test_result['recall'], 'Recall')
                show_topk_info(args, logger, test_result['ndcg'], 'NDCG')
                

                if test_result['recall'][2] >= best_recall_K_20:
                    best_recall_K_20 = test_result['recall'][2]
                    best_epoch = epoch + 1
                    auc_rec = test_result['auc']
                    ndcg_rec = test_result['ndcg'][2]
            
            logger.info('+++++++++++++')
            logger.info('Cur best epoch: <%.2d>, AUC: %.4f, Recall@20: %.4f, NDCG@20: %.4f'%(best_epoch, auc_rec, best_recall_K_20, ndcg_rec))
            logger.info('+++++++++++++')

            t_e_eval = time()
            eval_time = t_e_eval - t_s_eval
            tot_eval_time += eval_time
            eval_cnt += 1

            logger.info('Eval time: %.2f s'%(eval_time))
            logger.info('*******************************************************')

            cur_best_pre_0, stopping_step, should_stop = early_stopping(test_result['recall'][2], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)
            if should_stop:
                logger.info('Early stopping at %d with Recall@20:%.4f' % (epoch+1, cur_best_pre_0))
                break
    
    logger.info("================== Finish training ====================")
    logger.info('Best epoch: <%.2d>, AUC: %.4f, Recall@20: %.4f, NDCG@20: %.4f'%(best_epoch, auc_rec, best_recall_K_20, ndcg_rec))
    logger.info('Total training time: %.2f s, Avg Training Time: %.2f s (%d epochs), Total Eval time: %.2f s, Avg Eval Time: %.2f s (%d times)'
                %(tot_training_time, tot_training_time/(epoch+1), epoch+1, tot_eval_time, tot_eval_time/eval_cnt, eval_cnt))