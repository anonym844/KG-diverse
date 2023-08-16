import os
import sys
import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from model.model_div import KG_diverse
from parser.parser import *
from utils.log_helper import *
from utils.metrics_div import *
from utils.model_helper_div import *
from data_loader.loader_div import DataLoaderDIV

import pdb


def evaluate(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict
    item_entities = dataloader.item_entities
    item_relations = dataloader.item_relations
    edge_index = dataloader.edge_index
    edge_type = dataloader.edge_type
    # pdb.set_trace()

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = ['recall', 'ndcg', 'entity coverage', 'relation coverage']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}
    
    norm_item_embed = torch.nn.functional.normalize(model.generate(edge_index, edge_type)[1])

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                batch_scores = model.predict(batch_user_ids, item_ids,edge_index,edge_type)       # (n_batch_users, n_items)

            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(model, batch_scores, train_user_dict, test_user_dict, item_entities,item_relations, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks, norm_item_embed)

            cf_scores.append(batch_scores.numpy())
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return cf_scores, metrics_dict


def train(args,data):

    # GPU / CPU
    device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')

    # construct model & optimizer
    model = KG_diverse(args, data.n_users, data.n_items, data.n_entities, data.n_relations, data.A_in, data.Graph,data.item_sim_tensor,data.item_entities)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    edge_index = data.edge_index.to(device)
    edge_type = data.edge_type.to(device)

    best_val_loss = 99999999
    stop_count = 0
    # train model
    for epoch in range(1, args.n_epoch + 1):
        model.train()

        # train cf
        time1 = time()
        total_loss = 0

        n_batch = data.n_cf_train // data.cf_batch_size + 1
        val_batch = data.n_cf_val // data.cf_batch_size + 1

        for iter in range(1, n_batch + 1):

            batch_user, batch_pos_item, batch_neg_item = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            batch_user = batch_user.to(device)
            batch_pos_item = batch_pos_item.to(device)
            batch_neg_item = batch_neg_item.to(device)
            batch_loss = model(batch_user, batch_pos_item,batch_neg_item,edge_index,edge_type)

            if np.isnan(batch_loss.cpu().detach().numpy()):
                print('ERROR: Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_batch))
                sys.exit()

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss.item()

        val_loss = 0
        for iter in range(1, val_batch + 1):
            batch_user, batch_pos_item, batch_neg_item = data.generate_cf_batch(data.val_user_dict, data.cf_batch_size)
            batch_user = batch_user.to(device)
            batch_pos_item = batch_pos_item.to(device)
            batch_neg_item = batch_neg_item.to(device)
            batch_loss = model(batch_user, batch_pos_item,batch_neg_item,edge_index,edge_type)  
            val_loss += batch_loss.item()
        print('CF valid: Epoch {:04d} | Total Time {:.1f}s | Epoch Mean Loss {:.4f}'.format(epoch,time() - time1, val_loss / val_batch))


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_count = 0
            torch.save(model, os.path.join(args.save_dir, 'model_lr_{}_wd_{}_n-layers_{}_KG_layers_{}_gamma_{}_beta_{}.pt'.format(args.lr, args.wd, args.n_layers, args.KG_layers, args.gamma, args.beta)))
            # save_model(model, args)
        else:
            stop_count += 1
            if stop_count > args.stopping_steps:
                break


if __name__ == '__main__':
    args = parse_dpp_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    data = DataLoaderDIV(args)

    train(args,data)

    # predict
    device = torch.device('cpu')
    # model = KG_diverse(args, data.n_users, data.n_items, data.n_entities, data.n_relations)
    model = torch.load(os.path.join(args.save_dir, 'model_lr_{}_wd_{}_n-layers_{}_KG_layers_{}_gamma_{}_beta_{}.pt'.format(args.lr, args.wd, args.n_layers, args.KG_layers, args.gamma, args.beta)))
    model.to(device)

    # predict
    Ks = eval(args.Ks)

    cf_scores, metrics_dict = evaluate(model, data, Ks, device)
    print(metrics_dict)
    # predict(args)



