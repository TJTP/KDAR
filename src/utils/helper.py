import logging
import numpy as np 

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)

def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
 
    return n_params

def show_topk_info(args, logger, metrics, name):
    k_list = eval(args.k_list)
    topk_info = name + ': '
    for k, val in zip(k_list, metrics):
        topk_info += 'K@%d:%.4f '%(k, val)
    logger.info(topk_info)

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        logging.info("================== Early stop ====================")
        logging.info("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    
    return best_value, stopping_step, should_stop