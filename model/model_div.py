
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch_scatter import scatter_mean
import collections
from torch.utils.data import DataLoader
import os
import pickle
import random
# from sksos import SOS


class LightGCN(nn.Module):
    """
    LightGCN layer
    """
    def __init__(self, n_users, n_items, n_layers):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
    
    def forward(self, user_embeddings, item_embeddings, norm_adj_matrix):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(norm_adj_matrix, all_embeddings)
            
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings,
                                                               [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings
    
        
class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, n_items, temperature):
        super(Aggregator, self).__init__()
        self.n_items = n_items
        self.temperature = temperature

    def forward(self, entity_emb,
                edge_index, edge_type, 
                weight, interact_map,data_loader):

        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        # edge_relation_emb = weight[edge_type]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        # neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        
        neigh_relation_emb = entity_emb[tail]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        
        # entity_agg = entity_emb
        # pdb.set_trace()
        user_mean = torch.matmul(interact_map, entity_agg[:self.n_items])
        # return user_mean, entity_agg
        pdist = nn.PairwiseDistance(p=2)

        score = torch.tensor([]).to(entity_emb.device)
        for batch_ndx, sample in enumerate(data_loader):
            # pdb.set_trace()
            index,item = sample.t()
            score = torch.cat([score, pdist(entity_agg[item],user_mean[index])])

        index,item = interact_map.coalesce().indices()
        score_mat = torch.sparse.FloatTensor(torch.LongTensor([index.tolist(), item.tolist()]).to(entity_emb.device), score/self.temperature, size=interact_map.size())

        # soft_score_mat = torch.sparse.softmax(score_mat,dim=1).to_dense()
        soft_score_mat = torch.sparse.softmax(score_mat,dim=1)
        user_agg = torch.sparse.mm(soft_score_mat, entity_agg[:self.n_items])

        return user_agg, entity_agg


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,n_items,
                 n_relations, temperature,
                 node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()

        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.temperature = temperature
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations , channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        for i in range(n_hops):
            self.convs.append(Aggregator(self.n_items,self.temperature))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape)
        # random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]
        # out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        out = torch.sparse.FloatTensor(i, v, x.shape)
        return out * (1. / (1 - rate))

    def forward(self, user_emb, entity_emb,edge_index, edge_type,interact_map,dataloader,
                mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        for i in range(len(self.convs)):
            user_emb,entity_emb= self.convs[i](entity_emb,edge_index, edge_type,
                                                 self.weight,interact_map, dataloader)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)
        # return user_res_emb, entity_res_emb[:self.n_items]
        return user_res_emb, entity_res_emb


class KG_diverse(nn.Module):
    def __init__(self, args,n_users, n_items,n_entities,n_relations,interact_mat = None, Graph = None, item_sim_tensor = None, item_entities = None):
        super(KG_diverse, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_relations = n_relations
        self.n_entities = n_entities # include items

        self.emb_size = args.embed_dim
        self.context_hops = args.KG_layers
        self.gamma = args.gamma
        self.beta = args.beta
        self.temperature = args.temperature
        self.node_dropout = args.node_dropout
        self.node_dropout_rate = args.node_dropout_rate
        self.mess_dropout = args.mess_dropout
        self.mess_dropout_rate = args.mess_dropout_rate
        self.item_entities = item_entities
        # self.sim_layer = Similarity_layer(args.embed_dim*2,args.embed_dim,1)
    
        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(interact_mat)
        self.gcn = self._init_model()
        self.Graph = Graph
        self.lightgcn = LightGCN(self.n_users, self.n_items, args.n_layers)
        self.item_sim_tensor = item_sim_tensor
        # self.build_dict()
        # pdb.set_trace()
        self.dataloader()

    def dataloader(self):
        self.cf_dataloader = DataLoader(self.interact_mat.coalesce().indices().t(), batch_size=10000, shuffle=False)
        
    
    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_users+self.n_entities, self.emb_size))
        
    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_items=self.n_items,
                         n_relations=self.n_relations,
                         temperature = self.temperature,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def forward(self, user, pos_item, neg_item,edge_index, edge_type):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]

        user_gcn_emb, item_gcn_emb = self.gcn(user_emb,
                                item_emb,
                                edge_index,
                                edge_type,
                                self.interact_mat.to(self.all_embed.device),
                                self.cf_dataloader,
                                mess_dropout=self.mess_dropout,
                                node_dropout=self.node_dropout)
    
        
        # item_gcn_emb = entity_gcn_emb[:self.n_items]
        self.item_sim_tensor = self.item_sim_tensor.to(self.all_embed.device)
        
        
        KG_loss = self.conditional_align_uniform(item_gcn_emb, pos_item)
    
        # KG_loss = self.KG_loss(item_gcn_emb, pos_item)

        user_gcn_emb, item_gcn_emb = self.lightgcn(user_gcn_emb, item_gcn_emb[:self.n_items], self.Graph.to(self.all_embed.device))

        pos_e, neg_e = item_gcn_emb[:self.n_items][pos_item], item_gcn_emb[:self.n_items][neg_item]
        # pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        u_e = user_gcn_emb[user]

        cf_loss = self.create_bpr_loss(u_e, pos_e, neg_e)

        # loss = cf_loss + self.gamma * KG_loss
        loss = cf_loss + KG_loss

        return loss


    def predict(self, user, item, edge_index, edge_type):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        user_emb, item_emb = self.gcn(user_emb,
                        item_emb,
                        edge_index,
                        edge_type,
                        self.interact_mat.to(self.all_embed.device),
                        self.cf_dataloader,
                        mess_dropout=False, node_dropout=False)
        # item_emb = item_emb[:self.n_items]
        user_emb, item_emb = self.lightgcn(user_emb, item_emb[:self.n_items] , self.Graph.to(self.all_embed.device))
        # score = torch.matmul(user_emb[user], item_emb[item].t())
        score = torch.matmul(user_emb[user], item_emb[item].t())
        return score

    def generate(self, edge_index, edge_type):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # user_emb = self.user_embed
        # item_emb = self.entity_embed
        user_emb, item_emb = self.gcn(user_emb,
                item_emb,
                edge_index,
                edge_type,
                self.interact_mat.to(self.all_embed.device),
                self.cf_dataloader,
                mess_dropout=False, node_dropout=False)
        # item_emb = item_emb[:self.n_items]
        user_emb, item_emb = self.lightgcn(user_emb, item_emb[:self.n_items] , self.Graph.to(self.all_embed.device))
        return user_emb, item_emb
        # return self.gcn(user_emb,
        #                 item_emb,
        #                 edge_index,
        #                 edge_type,
        #                 mess_dropout=False, node_dropout=False)[:-1]
    
    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()


    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())


    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
        # mf_loss = -1* nn.LogSigmoid()(pos_scores - neg_scores)
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        return mf_loss


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
    
    def conditional_align_uniform(self, item_embed, batch_items):
        # alignment 
        batch_pos_item = []
        overlapping_embed_list = []
        for item in batch_items:
            sample_pos_item, sample_overlapping_embed = self.sample_pos_items_for_i(item, item_embed, 1)
            batch_pos_item += sample_pos_item
            overlapping_embed_list += sample_overlapping_embed
        batch_pos_item = torch.LongTensor(batch_pos_item).to(self.all_embed.device)
        item_embedding = item_embed[batch_items]
        pos_item_embed = item_embed[batch_pos_item]
        overlapping_embed = torch.stack(overlapping_embed_list)
        align_loss = self.alignment(item_embedding * overlapping_embed, pos_item_embed * overlapping_embed)
        # pdb.set_trace()
        
        # original alignment
        # batch_pos_item = []
        # overlapping_embed_list = []
        # for item in batch_items:
        #     sample_pos_item, sample_overlapping_embed = self.sample_pos_items_for_i(item, item_embed, 1)
        #     batch_pos_item += sample_pos_item
        #     overlapping_embed_list += sample_overlapping_embed
        # batch_pos_item = torch.LongTensor(batch_pos_item).to(self.all_embed.device)
        # item_embedding = item_embed[batch_items]
        # pos_item_embed = item_embed[batch_pos_item]
        # # overlapping_embed = torch.stack(overlapping_embed_list)
        # align_loss = self.alignment(item_embedding, pos_item_embed)
        
        # # projection alignment
        # batch_pos_item = []
        # overlapping_embed_list = []
        # for item in batch_items:
        #     sample_pos_item, sample_overlapping_embed = self.sample_pos_items_for_i(item, item_embed, 1)
        #     batch_pos_item += sample_pos_item
        #     overlapping_embed_list += sample_overlapping_embed
        # batch_pos_item = torch.LongTensor(batch_pos_item).to(self.all_embed.device)
        # item_embedding = item_embed[batch_items]
        # pos_item_embed = item_embed[batch_pos_item]
        # overlapping_embed = torch.stack(overlapping_embed_list)
        # x = ((item_embedding * overlapping_embed).sum(1) - (pos_item_embed * overlapping_embed).sum(1)).pow(2) / overlapping_embed.norm(p=2,dim=1)
        # align_loss = x.mean()
        # # KG_loss = self.gamma * align_loss
        
        # uniformity
        unique = torch.unique(batch_items)
        sample_items = torch.stack(random.sample(list(unique), 128)).to(self.all_embed.device)
        uniformity_loss = self.uniformity(item_embed[sample_items])
        KG_loss = self.gamma * align_loss + self.beta * uniformity_loss
        # KG_loss = self.gamma * (align_loss + self.beta * uniformity_loss)
        # KG_loss = self.gamma * align_loss
        return KG_loss
   
    
    def sample_pos_items_for_i(self, item_id, item_embed, n_sample_pos_items):
        pos_items = self.item_sim_tensor[item_id].nonzero()
        n_pos_items = len(pos_items)
        sample_pos_items = []
        overlapping = []
        if n_pos_items == 0:
           for i in range(n_sample_pos_items):
            #    pos_item_idx = np.random.randint(low=0, high=self.n_items, size=1)[0]
               sample_pos_items.append(item_id.item())
               overlapping.append(item_embed[item_id])
        while True:
            if len(sample_pos_items) == n_sample_pos_items:
                break

            pos_item_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_item_id = pos_items[pos_item_idx].item()
            if pos_item_id not in sample_pos_items:
                sample_pos_items.append(pos_item_id)
                overlap_entities = torch.LongTensor(list(set(self.item_entities[item_id.item()]).intersection(set(self.item_entities[pos_item_id]))))
                # mean
                overlapping_embed = item_embed[overlap_entities].mean(0)
                # # sample one
                # sample_entity_idx = np.random.randint(low=0, high=len(overlap_entities), size=1)[0]
                # sample_entity_id = overlap_entities[sample_entity_idx]
                # overlapping_embed = item_embed[sample_entity_id]
                
                overlapping.append(overlapping_embed)
        return sample_pos_items, overlapping
    
    # def KG_loss(self, item_embed, batch_items):
    #     batch_pos_item, batch_neg_item = [], []
    #     for item in batch_items:
    #         batch_pos_item += self.sample_pos_items_for_i(item, 1)
    #     batch_neg_item = torch.randint(self.n_items, (batch_items.shape[0],))
    #     batch_pos_item = torch.LongTensor(batch_pos_item).to(self.all_embed.device)
    #     batch_neg_item = torch.LongTensor(batch_neg_item).to(self.all_embed.device)
        
    #     index = self.item_sim_tensor[batch_items, batch_pos_item] - self.item_sim_tensor[batch_items, batch_neg_item]
    #     index[index>=0] = 1
    #     index[index<0] = -1
    #     item_embedding = item_embed[batch_items]
    #     pos_item_embed = item_embed[batch_pos_item]
    #     neg_item_embed = item_embed[batch_neg_item]
        
    #     # similarity layer
    #     # pos_score = self.sim_layer(torch.concat([item_embedding, pos_item_embed]))
    #     # neg_score = self.sim_layer(torch.concat([item_embedding, neg_item_embed]))
        
    #     # Cosine Sim
    #     pos_score = F.cosine_similarity(item_embedding, pos_item_embed)
    #     neg_score = F.cosine_similarity(item_embedding, neg_item_embed)
        
    #     # # coefficient
    #     # coefficient = []
    #     # for index in range(len(batch_items)):
    #     #     pos_overlap = len(set(self.item_entities[batch_items[index].item()]).union(set(self.item_entities[batch_pos_item[index].item()])))
    #     #     neg_overlap = len(set(self.item_entities[batch_items[index].item()]).union(set(self.item_entities[batch_neg_item[index].item()])))
    #     #     temp = abs( pos_overlap- neg_overlap) / max(pos_overlap, neg_overlap)
    #     #     coefficient.append(temp)
    #     # coefficient = torch.tensor(coefficient).to(self.all_embed.device)
    #     # KG_loss = -1 * torch.mean(coefficient * nn.LogSigmoid()(index*(pos_score - neg_score)))
        
    #     # # L2-distancce
    #     # pdist = nn.PairwiseDistance(p=2)
    #     # pos_score = pdist(item_embedding, pos_item_embed)
    #     # neg_score = pdist(item_embedding, neg_item_embed)
        
    #     # KG_loss = -1 * torch.mean(nn.LogSigmoid()(index*(pos_score - neg_score)))
        
    #     # alignment
    #     KG_loss = self.alignment(item_embedding, pos_item_embed)
        
    #     return KG_loss

    # def sample_pos_items_for_i(self, item_id, n_sample_pos_items):
    #     pos_items = self.item_sim_tensor[item_id].nonzero()
    #     # pdb.set_trace()
    #     n_pos_items = len(pos_items)
    #     sample_pos_items = []
    #     if n_pos_items == 0:
    #        for i in range(n_sample_pos_items):
    #            pos_item_idx = np.random.randint(low=0, high=self.n_items, size=1)[0]
    #            sample_pos_items.append(pos_item_idx)
    #     while True:
    #         if len(sample_pos_items) == n_sample_pos_items:
    #             break

    #         pos_item_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
    #         pos_item_id = pos_items[pos_item_idx].item()
    #         if pos_item_id not in sample_pos_items:
    #             sample_pos_items.append(pos_item_id)
    #     return sample_pos_items
    

class Similarity_layer(torch.nn.Module):
    def __init__(self, in_size, layer_size, out_size):
        super(Similarity_layer, self).__init__()
        self.linear1 = torch.nn.Linear(in_size, layer_size)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(layer_size, layer_size)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(layer_size, out_size)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x