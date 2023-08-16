import os
from collections import OrderedDict

import torch


def early_stopping(recall_list, stopping_steps):
    best_recall = max(recall_list)
    best_step = recall_list.index(best_recall)
    if len(recall_list) - best_step - 1 >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    return best_recall, should_stop


def save_model(model, args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model_state_file = os.path.join(args.save_dir, 'model_lr_{}_wd_{}_n-layers_{}_KG_layers_{}_gamma_{}_beta_{}.pt'.format(args.lr, args.wd, args.n_layers, args.KG_layers, args.gamma, args.beta))
    torch.save({'model_state_dict': model.state_dict()}, model_state_file)

    # if last_best_epoch is not None and current_epoch != last_best_epoch:
    #     old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
    #     if os.path.exists(old_model_state_file):
    #         os.system('rm {}'.format(old_model_state_file))


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


