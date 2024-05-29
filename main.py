import os
import random

import torch
import numpy as np
from time import time


from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test

import optuna
import joblib
import datetime
from sklearn.metrics import roc_auc_score,average_precision_score


n_genes = 0
n_drugs = 0


def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n_negs=1, K=1, n_drugs=0):
    def sampling(gene_item, train_set, n):
        neg_items = []
        for user, _ in gene_item.cpu().numpy():
            user = int(user)
            negitems = []
            for i in range(n):  # sample n times
                while True:
                    negitem = random.choice(range(n_drugs))
                    if negitem not in train_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['genes'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                       train_pos_set,
                                                       n_negs * K)).to(device)
    return feed_dict


def opt_objective(trial, args, train_cf, gene_dict, n_params, norm_mat, deg, outdeg):
    valid_res_list = []

    # args.dim = trial.suggest_int('dim', 16, 512)
    args.l2 = trial.suggest_float('l2', 0, 1)
    args.context_hops = trial.suggest_int('context_hops', 1, 6)

    for seed in range(args.runs):
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(args)
        valid_best_result = main(args, seed, train_cf, gene_dict, n_params, norm_mat, deg, outdeg)
        valid_res_list.append(valid_best_result)
    return np.mean(valid_res_list)


def main(args, run, train_cf, gene_dict, n_params, norm_mat, deg, outdeg):
    """define model"""
    from model import HeatKernel


    model = HeatKernel(n_params, args, norm_mat, deg).to(device)

    
    """define optimizer"""
    optimizer = torch.optim.Adam([{'params': model.parameters(),
                                   'lr': args.lr}])
    n_drugs = n_params['n_drugs']
    cur_best_pre_0 = 0
    best_epoch = 0
    print("start training ...")

    hyper = {"dim": args.dim, "l2": args.l2, "hops": args.context_hops}
    print("Start hyper parameters: ", hyper)
    best_auc = 0
    best_aupr = 0
    for epoch in range(args.epoch):
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)

        """training"""
        model.train()
        loss, s = 0, 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_,
                                  gene_dict['train_gene_set'],
                                  s, s + args.batch_size,
                                  args.n_negs,
                                  args.K,
                                  n_drugs)
            batch_loss, _, _ = model(batch)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size
        train_e_t = time()
        print('loss:', round(loss.item(), 2), "time: ", round(train_e_t - train_s_t, 2), 's')

    print("End hyper parameters: ", hyper)
    return cur_best_pre_0


if __name__ == '__main__':
    """read args"""
    global args, device
    args = parse_args()
    s = datetime.datetime.now()
    print("time of start: ", s)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    """build dataset"""
    train_cf, gene_dict, n_params, norm_mat, deg, outdeg = load_data(args)
    train_cf_size = len(train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    trials = 1
    search_space = {'dim': [512], 'context_hops': [2], 'l2': [1e-3]}
    print("search_space: ", search_space)
    print("trials: ", trials)
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(lambda trial: opt_objective(trial, args, train_cf, gene_dict, n_params, norm_mat, deg, outdeg), n_trials=trials)
    joblib.dump(study,
                f'{args.dataset}_{args.dim}_{args.context_hops}_{args.l2}_study_' + args.gnn + '.pkl')
    e = datetime.datetime.now()
    print(study.best_trial.params)
    print("time of end: ", e)
