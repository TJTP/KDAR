import os
import math
import numpy as np 
import scipy.sparse as sp
from collections import defaultdict


class Dataset(object):
    def __init__(self, args, logger):
        self.name = args.dataset
        
        logger.info("================== Loading data: %s ... ==================="%(args.dataset))
        self._load_collaborative_data(args, logger)
        self._load_kg_triplets(args, logger)
        
        self._construct_adj_matrices(args, logger)
        
        self.n_nodes = self.n_users + self.n_entities
    
    def _construct_user_record(self, interactions):
            user_record = defaultdict(set)
            for inter in interactions:
                user, item, rating = inter
                if rating == 1:
                    user_record[user].add(item)
            return user_record
        
    def _load_collaborative_data(self, args, logger):
        logger.info("Loading collaborative files...")
        
        data_dir = os.path.join(args.data_dir, args.dataset)

        self.train_data = np.loadtxt(os.path.join(data_dir, 'train_pos.txt'), dtype=np.int32)
        self.test_data  = np.loadtxt(os.path.join(data_dir, 'test.txt'),      dtype=np.int32)
        
        self.item_set = set(self.train_data[:, 1]).union(self.test_data[:, 1])

        self.train_user_record = self._construct_user_record(self.train_data)
        self.test_user_record  = self._construct_user_record(self.test_data)

        self.user_candidate_items = dict()
        for user in self.train_user_record.keys():
            if user not in self.user_candidate_items.keys():
                self.user_candidate_items[user] = list(self.item_set - self.train_user_record[user])
            else:
                assert "Multiple users!"
        
        self.n_users = max(max(self.train_data[:, 0]), max(self.test_data[:, 0])) + 1
        self.n_items = max(max(self.train_data[:, 1]), max(self.test_data[:, 1])) + 1

    def _load_kg_triplets(self, args, logger):
        logger.info('Loading KG triplets...')

        kg_triplet_array = np.loadtxt(os.path.join(args.data_dir, args.dataset, 'kg_final.txt'), dtype=np.int32)
        
        if args.inverse_r:
            n_original_relations = max(kg_triplet_array[:, 1]) + 1

            inv_triplets = kg_triplet_array.copy()
            inv_triplets[:, 0] = kg_triplet_array[:, 2]
            inv_triplets[:, 2] = kg_triplet_array[:, 0]
            inv_triplets[:, 1] = kg_triplet_array[:, 1] + n_original_relations
            kg_triplet_array = np.concatenate((kg_triplet_array, inv_triplets), axis=0)
        
        kg_triplet_array = np.unique(kg_triplet_array, axis=0)

        self.n_entities = max(max(kg_triplet_array[:, 0]), max(kg_triplet_array[:, 2])) + 1   
        self.n_relations = max(kg_triplet_array[:, 1]) + 1 # including inverse relations
        self.n_kg_triplets = len(kg_triplet_array) # including inverse triplets

        self._triplet_array = kg_triplet_array.T

    def _construct_adj_matrices(self, args, logger):
        def _si_norm_lap(adj):
            # D^{-1}A
            rowsum = np.array(adj.sum(1))
            
            rowsum_inv = np.power(rowsum, -1.0).flatten()
            rowsum_inv[np.isinf(rowsum_inv)] = 0.
            rowsum_inv = np.reshape(rowsum_inv, [-1, 1])
            
            norm_mat = adj.multiply(rowsum_inv)

            return norm_mat
        
        inter_data = self.train_data.copy()
        user_ids, item_ids, values = inter_data[:, 0], inter_data[:, 1], inter_data[:, 2]
        
        inter_mat = sp.coo_matrix((values, (user_ids, item_ids)), 
                                  shape=(self.n_users, self.n_items)) # [n_users, n_items]
        inter_mat_trans = inter_mat.transpose() # [n_items, n_users]

        user_neighbors_num = np.array(inter_mat.sum(axis=1)) # [n_users, ]
        item_neighbors_num = np.array(inter_mat_trans.sum(axis=1)) # [n_items, ]

        item_neighbors_num_inv = np.power(item_neighbors_num, -1.0).flatten()
        item_neighbors_num_inv[np.isinf(item_neighbors_num_inv)] = 0.
        item_neighbors_num_inv = np.reshape(item_neighbors_num_inv, [1, -1]) # [1, n_items]

        user_neighbors_num_inv = np.power(user_neighbors_num, -1.0).flatten()
        user_neighbors_num_inv[np.isinf(user_neighbors_num_inv)] = 0.
        user_neighbors_num_inv = np.reshape(user_neighbors_num_inv, [-1, 1]) # [n_users, 1]

        user_adj_mat = inter_mat.multiply(np.sqrt(item_neighbors_num_inv)).multiply(np.sqrt(user_neighbors_num_inv))
        
        self.user_adj_mat = user_adj_mat
        self.item_adj_mat = user_adj_mat.transpose()

        # degree adj mat
        self.inter_mat = _si_norm_lap(inter_mat)

