import numpy as np
import scipy.sparse as sp

from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

n_genes = 0
n_drugs = 0
dataset = ''
train_gene_set = defaultdict(list)
train_drug_set = defaultdict(list)
test_gene_set = defaultdict(list)
valid_gene_set = defaultdict(list)



def read_cf_yelp2018(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        g_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for d_id in pos_ids:
            inter_mat.append([g_id, d_id])
    return np.array(inter_mat)


def statistics(train_data, valid_data, test_data):
    global n_genes, n_drugs
    n_genes = max(max(train_data[:, 0]), max(valid_data[:, 0]), max(test_data[:, 0])) + 1
    n_drugs = max(max(train_data[:, 1]), max(valid_data[:, 1]), max(test_data[:, 1])) + 1

    if dataset not in ['dgidb5']:
        n_drugs -= n_genes
        # remap [n_genes, n_genes+n_drugs] to [0, n_drugs]
        train_data[:, 1] -= n_genes
        valid_data[:, 1] -= n_genes
        test_data[:, 1] -= n_genes

    cnt_train, cnt_test, cnt_valid = 0, 0, 0
    for g_id, d_id in train_data:
        train_gene_set[int(g_id)].append(int(d_id))
        train_drug_set[int(d_id)].append(int(g_id))
        cnt_train += 1
    for g_id, d_id in test_data:
        test_gene_set[int(g_id)].append(int(d_id))
        cnt_test += 1
    for g_id, d_id in valid_data:
        valid_gene_set[int(g_id)].append(int(d_id))
        cnt_valid += 1
    print('n_genes: ', n_genes, '\tn_drugs: ', n_drugs)
    print('n_train: ', cnt_train, '\tn_test: ', cnt_test, '\tn_valid: ', cnt_valid)
    print('n_inters: ', cnt_train + cnt_test + cnt_valid)


def build_sparse_graph(data_cf):
    def _bd_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bd_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bd_lap.tocoo()

    def _sd_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    cf = data_cf.copy()
    cf[:, 1] = cf[:, 1] + n_genes  # [0, n_drugs) -> [n_genes, n_genes+n_drugs)
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

    # diag = np.array([[i, i] for i in range(n_genes+n_drugs)])
    # cf_ = np.concatenate([cf, cf_, diag], axis=0)  # [[0, R], [R^T, 0]] + I
    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

    vals = [1.] * len(cf_)
    mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_genes + n_drugs, n_genes + n_drugs))
    return _bd_norm_lap(mat), np.array(mat.sum(1)), np.array(mat.sum(0))


def load_data(model_args):
    global args, dataset
    args = model_args
    dataset = args.dataset
    directory = args.data_path + dataset + '/'

    if  dataset == 'dgidb5':
        read_cf = read_cf_yelp2018

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    valid_cf = test_cf
    statistics(train_cf, valid_cf, test_cf)

    print('building the adj mat ...')
    norm_mat, indeg, outdeg = build_sparse_graph(train_cf)

    n_params = {
        'n_genes': int(n_genes),
        'n_drugs': int(n_drugs),
    }
    gene_dict = {
        'train_drug_set': train_drug_set,
        'train_gene_set': train_gene_set,
        'valid_gene_set': None,
        'test_gene_set': test_gene_set,
    }
    print('loading over ...')
    return train_cf, gene_dict, n_params, norm_mat, indeg, outdeg
