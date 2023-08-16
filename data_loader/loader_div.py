import os
import random
import collections

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle
from data_loader.loader_base import DataLoaderBase
import pdb
from scipy.sparse import csr_matrix


class DataLoaderDIV(DataLoaderBase):

    def __init__(self, args):
        super().__init__(args)
        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size

        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)
        self.print_info()
        # self.create_adjacency_dict()
        self.create_laplacian_dict()
        self.construct_sim(self.data_dir)
    
    def construct_sim(self, sim_path):
        def jaccard_sim(mat, n_items):
            co_share = torch.matmul(mat,mat.t())
            diag = torch.diag(co_share, 0)
            diag_reshape = torch.reshape(diag,(diag.shape[0],1))
            expand = diag_reshape.expand(diag_reshape.shape[0],n_items)
            union = expand+diag
            result = (co_share/(union - co_share))
            return result
        
        if not os.path.isfile(sim_path+'/item_sim.t'):
            print("generating item similarity tensor")
            adj = torch.zeros(self.n_items, self.n_entities)

            for k,v in self.item_entities.items():
                for i in v:
                    adj[k][i] = 1.0
                    
            item_sim_tensor = jaccard_sim(adj, self.n_items)
            item_sim_tensor = torch.nan_to_num(item_sim_tensor)
            torch.save(item_sim_tensor, sim_path+'/item_sim.t')
        item_sim_tensor = torch.load(sim_path+'/item_sim.t')
        print("successfully loaded item similarity tensor")
        self.item_sim_tensor = item_sim_tensor


    def construct_data(self, kg_data):

        self.n_relations = max(kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        self.n_users_entities = self.n_users + self.n_entities

        self.kg_train_data = kg_data
        self.n_kg_train = len(self.kg_train_data)
        # construct kg dict
        h_list = []
        t_list = []
        r_list = []

        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for row in self.kg_train_data.iterrows():
            h, r, t = row[1]
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)

            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))
        self.edge_index = torch.stack([torch.LongTensor(h_list),torch.LongTensor(t_list)])
        self.edge_type = torch.LongTensor(r_list)
        # self.h_list = torch.LongTensor(h_list)
        # self.t_list = torch.LongTensor(t_list)
        # self.r_list = torch.LongTensor(r_list)


    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))


    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()
        def _si_norm_lap(adj):
            # D^{-1}A
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()
        norm_lap_func = _si_norm_lap
        rows, cols = self.cf_train_data
        
        cf0 = rows.copy()
        cf1 = cols.copy()
        cf1 = cf1 + self.n_users  # [0, n_items) -> [n_users, n_users+n_items)
        vals = [1.] * len(cf1)
        adj = sp.coo_matrix((vals, (cf0, cf1)), shape=(self.n_users_entities, self.n_users_entities))
        mean_mat_list = norm_lap_func(adj)
        self.A_in = mean_mat_list.tocsr()[:self.n_users,self.n_users:self.n_users+self.n_items].tocoo()

        try:
            pre_adj_mat = sp.load_npz(self.data_dir + '/s_pre_adj_mat.npz')
            print("successfully loaded adjacency matrix")
            norm_adj = pre_adj_mat
        except:
            print("generating adjacency matrix")
            UserItemNet = csr_matrix((np.ones(len(rows)), (rows, cols)),
                                        shape=(self.n_users, self.n_items))
            adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = UserItemNet.tolil()
            adj_mat[:self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, :self.n_users] = R.T
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
            
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            sp.save_npz(self.data_dir + '/s_pre_adj_mat.npz', norm_adj)
        self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj).coalesce()

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def print_info(self):
        print('n_users:           %d' % self.n_users)
        print('n_items:           %d' % self.n_items)
        print('n_entities:        %d' % self.n_entities)
        # print('n_users_entities:  %d' % self.n_users_entities)
        print('n_relations:       %d' % self.n_relations)
        print('n_triplets:        %d' % len(self.edge_type))

        print('n_cf_train:        %d' % self.n_cf_train)
        print('n_cf_test:         %d' % self.n_cf_test)

        print('n_kg_train:        %d' % self.n_kg_train)

