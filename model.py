import torch
import torch.nn as nn
from scipy.special import factorial, comb
import scipy.sparse as sp


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, n_hops, n_genes, n_drugs, interact_mat, deg,
                 e=1e-7, edge_dropout_rate=0.5, mess_dropout_rate=0.1,
                 t_g=2, t_d=2, device=6):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.n_genes = n_genes
        self.n_drugs = n_drugs
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout
        self.dev = device
        # set (n_genes+n_drugs) t
        initializer = nn.init.xavier_uniform_
        self.gene_t = initializer(torch.empty(self.n_genes, 1))
        self.drug_t = initializer(torch.empty(self.n_drugs, 1))
        self.gene_t = nn.Parameter(self.gene_t, requires_grad=True)
        self.drug_t = nn.Parameter(self.drug_t, requires_grad=True)
        self.degree = torch.sigmoid(torch.log(torch.from_numpy(deg) + e))
        t_g, t_d = torch.split(self.degree, [self.n_genes, self.n_drugs], dim=0)
        self.gene_t.data = t_g
        self.drug_t.data = t_d

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, gene_embed, drug_embed, mess_dropout=True, edge_dropout=True):
        all_embed = torch.cat([gene_embed, drug_embed], dim=0)
        agg_embed = all_embed
        g_weight = (torch.exp(-self.gene_t).to(self.dev) * torch.pow(self.gene_t, torch.FloatTensor([0]).to(self.dev))
                    .to(self.dev) / torch.FloatTensor([factorial(0)]).to(self.dev))
        d_weight = (torch.exp(-self.drug_t).to(self.dev) * torch.pow(self.drug_t, torch.FloatTensor([0]).to(self.dev))
                    .to(self.dev) / torch.FloatTensor([factorial(0)]).to(self.dev))
        ego_embeddings = torch.cat([g_weight * gene_embed, d_weight * drug_embed], dim=0)
        embs = [ego_embeddings]

        for k in range(1, self.n_hops + 1):
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                self.edge_dropout_rate) if edge_dropout \
                else self.interact_mat

            side_embeddings = torch.sparse.mm(interact_mat, agg_embed)
            # agg_embed = F.normalize(agg_embed)
            gene_embedds, drug_embedds = torch.split(side_embeddings, [self.n_genes, self.n_drugs],
                                                     dim=0)
            gene_embedds = gene_embedds * (torch.exp(-self.gene_t).to(self.dev) *
                                           torch.pow(self.gene_t, torch.FloatTensor([k]).to(self.dev)).to(self.dev)
                                           / torch.FloatTensor([factorial(k)]).to(self.dev))
            drug_embedds = drug_embedds * (torch.exp(-self.drug_t).to(self.dev) *
                                           torch.pow(self.drug_t, torch.FloatTensor([k]).to(self.dev)).to(self.dev)
                                           / torch.FloatTensor([factorial(k)]).to(self.dev))
            side_embeddings_cur = torch.cat([gene_embedds, drug_embedds], dim=0)
            agg_embed = side_embeddings
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            embs.append(side_embeddings_cur)
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        return embs[:self.n_genes, :], embs[self.n_genes:, :]


class HeatKernel(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, deg):
        super(HeatKernel, self).__init__()

        self.n_genes = data_config['n_genes']
        self.n_drugs = data_config['n_drugs']
        self.adj_mat = adj_mat
        self.deg = deg
        self.e = args_config.e
        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.edge_dropout = args_config.edge_dropout
        self.edge_dropout_rate = args_config.edge_dropout_rate
        self.pool = args_config.pool
        self.n_negs = args_config.n_negs
        self.ns = args_config.ns
        self.K = args_config.K
        self.heatkernel = args_config.heatkernel

        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")
        self.t_g = args_config.t_g
        self.t_d = args_config.t_d
        self._init_weight()
        self.gene_embed = nn.Parameter(self.gene_embed)
        self.drug_embed = nn.Parameter(self.drug_embed)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.gene_embed = initializer(torch.empty(self.n_genes, self.emb_size))
        self.drug_embed = initializer(torch.empty(self.n_drugs, self.emb_size))

        # [n_genes+n_drugs, n_genes+n_drugs]
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_genes=self.n_genes,
                         n_drugs=self.n_drugs,
                         interact_mat=self.sparse_norm_adj,
                         deg=self.deg,
                         e=self.e,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate,
                         t_g=self.t_g,
                         t_d=self.t_d,
                         device=self.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, batch=None):
        user = batch['genes']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']  # [batch_size, n_negs * K]

        genes_gcn_emb, drug_gcn_emb = self.gcn(self.gene_embed,
                                              self.drug_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)

        if self.ns == 'rns':  # n_negs = 1
            neg_gcn_embs = drug_gcn_emb[neg_item[:, :self.K]]
        else:
            neg_gcn_embs = []
            for k in range(self.K):
                neg_gcn_embs.append(self.negative_sampling(genes_gcn_emb, drug_gcn_emb,
                                                           user, neg_item[:, k * self.n_negs: (k + 1) * self.n_negs],
                                                           pos_item))
            neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)

        return self.create_bpr_loss(genes_gcn_emb[user], drug_gcn_emb[pos_item], neg_gcn_embs)

    def negative_sampling(self, genes_gcn_emb, drug_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = genes_gcn_emb[user], drug_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1)

        """positive mixing"""
        seed = torch.rand(batch_size, 1, p_e.shape[1], 1).to(p_e.device)  # (0, 1)
        n_e = drug_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops, channel]
        n_e_ = seed * p_e.unsqueeze(dim=1) + (1 - seed) * n_e  # mixing

        """hop mixing"""
        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        indices = torch.max(scores, dim=1)[1].detach()
        neg_items_emb_ = n_e_.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        # [batch_size, n_hops+1, channel]
        return neg_items_emb_[[[i] for i in range(batch_size)],
               range(neg_items_emb_.shape[1]), indices, :]

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def generate(self, split=True):
        genes_gcn_emb, drug_gcn_emb = self.gcn(self.gene_embed,
                                              self.drug_embed,
                                              edge_dropout=False,
                                              mess_dropout=False)
        genes_gcn_emb, drug_gcn_emb = self.pooling(genes_gcn_emb), self.pooling(drug_gcn_emb)
        if split:
            return genes_gcn_emb, drug_gcn_emb
        else:
            return torch.cat([genes_gcn_emb, drug_gcn_emb], dim=0)

    def rating(self, g_g_embeddings=None, d_g_embeddings=None):
        return torch.matmul(g_g_embeddings, d_g_embeddings.t())

    def create_bpr_loss(self, genes_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        # genes_gcn_emb: [batch_size, n_hops+1, channel]
        # pos_gcn_embs: [batch_size, n_hops+1, channel]
        # neg_gcn_embs: [batch_size, K, n_hops+1, channel]

        batch_size = genes_gcn_emb.shape[0]

        g_e = self.pooling(genes_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(batch_size,
                                                                                                       self.K, -1)

        pos_scores = torch.sum(torch.mul(g_e, pos_e), axis=1)
        neg_scores = torch.sum(torch.mul(g_e.unsqueeze(dim=1), neg_e), axis=-1)  # [batch_size, K]

        mf_loss = torch.mean(torch.log(1 + torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)))

        # cul regularizer
        regularize = (torch.norm(genes_gcn_emb[:, 0, :]) ** 2
                      + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
                      + torch.norm(neg_gcn_embs[:, :, 0, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss