from .parser import parse_args

import random
import torch
import math
import numpy as np
import multiprocessing
import heapq
from time import time

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)
device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag
gene_recall_result = dict()
deg_recall = dict()
deg_recall_mean = dict()





def test(model, gene_dict, n_params, deg, mode='test'):

    global n_genes, n_drugs, gene_recall_result, deg_recall, deg_recall_mean
    n_drugs = n_params['n_drugs']
    n_genes = n_params['n_genes']

    global train_gene_set, test_gene_set
    train_gene_set = gene_dict['train_gene_set']
    if mode == 'test':
        test_gene_set = gene_dict['test_gene_set']
    else:
        test_gene_set = gene_dict['valid_gene_set']
        if test_gene_set is None:
            test_gene_set = gene_dict['test_gene_set']


    g_batch_size = BATCH_SIZE
    d_batch_size = BATCH_SIZE

    test_genes = list(test_gene_set.keys())
    n_test_genes = len(test_genes)
    n_gene_batchs = n_test_genes // g_batch_size + 1

    count = 0

    genes_gcn_emb, drug_gcn_emb = model.generate()
    pred_score = model.rating(genes_gcn_emb,drug_gcn_emb).detach().cpu().numpy().tolist()
    y_true = []
    y_pred = []
    for g_batch_id in range(n_gene_batchs):
        start = g_batch_id * g_batch_size
        end = (g_batch_id + 1) * g_batch_size

        gene_list_batch = test_genes[start:end]
        gene_batch = torch.LongTensor(np.array(gene_list_batch)).to(device)
        g_g_embeddings = genes_gcn_emb[gene_batch]

        if batch_test_flag:
            # batch-item test
            n_drug_batchs = n_drugs // d_batch_size + 1
            rate_batch = np.zeros(shape=(len(gene_batch), n_drugs))

            d_count = 0
            for d_batch_id in range(n_drug_batchs):
                d_start = d_batch_id * d_batch_size
                d_end = min((d_batch_id + 1) * d_batch_size, n_drugs)

                drug_batch = torch.LongTensor(np.array(range(d_start, d_end))).view(d_end-d_start).to(device)
                d_g_embddings = drug_gcn_emb[drug_batch]

                d_rate_batch = model.rating(g_g_embeddings, d_g_embddings).detach().cpu()

                rate_batch[:, d_start:d_end] = d_rate_batch
                d_count += d_rate_batch.shape[1]

            assert d_count == n_drugs
        else:
            # all-item test
            drug_batch = torch.LongTensor(np.array(range(0, n_drugs))).view(n_drugs).to(device)
            d_g_embddings = drug_gcn_emb[drug_batch]
            rate_batch = model.rating(g_g_embeddings, d_g_embddings).detach().cpu()

        for i in range(len(gene_list_batch)):
            uid = gene_list_batch[i]
            drug_scores = rate_batch[i]
            pos = test_gene_set[uid]
            train_drug_ass = train_gene_set[uid]
            train_test_ass = pos + train_drug_ass
            diff = list(set(range(len(drug_scores))) - set(train_test_ass))
            random.shuffle(diff)
            neg = diff[0:len(pos)]
            y_true += [1] * len(pos)
            y_true += [0] * len(pos)
            for item in pos:
                y_pred.append(drug_scores[item])
            for item in neg:
                y_pred.append(drug_scores[item])
    return  y_true, y_pred, pred_score
