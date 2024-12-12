import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import time
import numpy as np
import scipy.sparse as sp
from copy import deepcopy

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from utils import accuracy

from PGD import PGD, prox_operators

class FALayer(MessagePassing):
    def __init__(self, labels, adj, num_hidden, dropout):
        super(FALayer, self).__init__(aggr='add')
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * num_hidden, 1)
        self.device = adj.device
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)


    def forward(self, h, adj_hom, adj_het):
        self.adj_hom = adj_hom.to(self.device)
        self.adj_het = adj_het.to(self.device)
        self.row_hom = torch.tensor(sp.coo_matrix(adj_hom.cpu().detach()).row, dtype=torch.int64).to(self.device)
        self.col_hom = torch.tensor(sp.coo_matrix(adj_hom.cpu().detach()).col, dtype=torch.int64).to(self.device)
        self.edge_index_hom = torch.vstack((self.row_hom, self.col_hom))
        self.norm_degree_hom = torch.sum(self.adj_hom, dim=0).detach()
        self.norm_degree_hom = torch.pow(self.norm_degree_hom, -0.5)
        self.norm_degree_hom[torch.isinf(self.norm_degree_hom)] = 0.
        self.row_het = torch.tensor(sp.coo_matrix(adj_het.cpu().detach()).row, dtype=torch.int64).to(self.device)
        self.col_het = torch.tensor(sp.coo_matrix(adj_het.cpu().detach()).col, dtype=torch.int64).to(self.device)
        self.edge_index_het = torch.vstack((self.row_het, self.col_het))
        self.norm_degree_het = torch.sum(self.adj_het, dim=0).detach()
        self.norm_degree_het = torch.pow(self.norm_degree_het, -0.5)
        self.norm_degree_het[torch.isinf(self.norm_degree_het)] = 0.

        h_hom = torch.cat([h[self.row_hom], h[self.col_hom]], dim=1)
        g_hom = torch.tanh(self.gate(h_hom)).squeeze()
        h_het = torch.cat([h[self.row_het], h[self.col_het]], dim=1)
        g_het = torch.tanh(self.gate(h_het)).squeeze()

        norm_hom = g_hom * self.norm_degree_hom[self.row_hom] * self.norm_degree_hom[self.col_hom] * self.adj_hom[self.row_hom, self.col_hom]
        norm_hom = self.dropout(norm_hom)
        norm_het = g_het * self.norm_degree_het[self.row_het] * self.norm_degree_het[self.col_het] * self.adj_het[self.row_het, self.col_het]
        norm_het = self.dropout(norm_het)
        return (self.propagate(self.edge_index_hom, size=(h.size(0), h.size(0)), x=h, norm=norm_hom)/2 +
                self.propagate(self.edge_index_het, size=(h.size(0), h.size(0)), x=h, norm=norm_het)/2)

    def message(self, x_j, norm):
        return norm.view(-1,1) * x_j

    def update(self, aggr_out):
        return aggr_out




class FAGCN_wodgl(nn.Module):
    def __init__(self, labels, adj, num_features, num_hidden, num_classes, dropout, eps, layer_num=2):
        super(FAGCN_wodgl, self).__init__()
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(labels, adj, num_hidden, dropout))
        self.t1 = nn.Linear(num_features, num_hidden)
        self.t2 = nn.Linear(num_hidden, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h, adj_hom, adj_het):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h, adj_hom, adj_het)
            h = self.eps * raw + h
        h = self.t2(h)
        return F.log_softmax(h, 1)



class SplitGNN:

    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.best_graph_hom = None
        self.best_graph_het = None


    def fit(self, features, adj, labels, idx_train, idx_val, idx_test):
        self.optimizer_gnn = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                          weight_decay=self.args.weight_decay)
        num_of_nodes = features.shape[0]
        A1 = EstimateAdj(adj, num_of_nodes, hom=True, symmetric=False, device=self.args.device).to(self.args.device)
        A2 = EstimateAdj(adj, num_of_nodes, hom=False, symmetric=False, device=self.args.device).to(self.args.device)
        if self.args.dataset in ['cora', 'citeseer', 'pubmed']:
            self.estimator_adj_hom = A1
            self.estimator_adj_het = A2
        else:
            self.estimator_adj_hom = A2
            self.estimator_adj_het = A1
        self.optimizer_adj_hom = torch.optim.SGD(self.estimator_adj_hom.parameters(),
                              momentum=0.9, lr=self.args.lr_adj)
        self.optimizer_adj_het = torch.optim.SGD(self.estimator_adj_het.parameters(),
                              momentum=0.9, lr=self.args.lr_adj)

        self.optimizer_l1_hom = PGD(self.estimator_adj_hom.parameters(), proxs=[prox_operators.prox_l1], lr=self.args.lr_adj, alphas=[self.args.alpha])
        self.optimizer_l1_het = PGD(self.estimator_adj_het.parameters(), proxs=[prox_operators.prox_l1], lr=self.args.lr_adj, alphas=[self.args.alpha])
        self.optimizer_nuclear_hom = PGD(self.estimator_adj_hom.parameters(),
                  proxs=[prox_operators.prox_nuclear_cuda],
                  lr=self.args.lr_adj, alphas=[self.args.beta])
        self.optimizer_nuclear_het = PGD(self.estimator_adj_het.parameters(),
                                     proxs=[prox_operators.prox_nuclear_cuda],
                                     lr=self.args.lr_adj, alphas=[self.args.beta])

        self.dur = []
        self.los = []
        self.loc = []
        self.counter = 0
        self.min_loss = 100.0
        self.max_acc = 0.0
        self.best_epoch = 0

        for epoch in range(self.args.epochs):

            if epoch >= 3:
                t0 = time.time()

            for i in range(int(self.args.train_adj_steps)):
                self.train_adj(
                    epoch, features, labels, idx_train, idx_val, idx_test, adj)
            for j in range(int(self.args.train_gnn_steps)):
                loss_val, train_acc, val_acc, test_acc = self.train_gnn(
                    epoch, features, labels, idx_train, idx_val, idx_test)

            if loss_val < self.min_loss and self.max_acc < val_acc:
                self.min_loss = loss_val
                self.max_acc = val_acc
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1

            if self.counter >= self.args.patience and self.args.dataset in ['cora', 'citeseer', 'pubmed']:
                print('early stop')
                break

            if epoch >= 3:
                self.dur.append(time.time() - t0)
            print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f} | Time(s) {:.4f}".format(
                epoch, loss_val, train_acc, val_acc, test_acc, np.mean(self.dur)))


        if self.args.dataset in ['cora', 'citeseer', 'pubmed'] or 'syn' in self.args.dataset:
            self.los.sort(key=lambda x: x[1])
            acc = self.los[0][-1]
            print(acc)

        else:
            self.los.sort(key=lambda x: -x[2])
            acc = self.los[0][-1]
            print(acc) 

        return acc

    def train_gnn(self, epoch, features, labels, idx_train, idx_val, idx_test):
        adj_hom = self.estimator_adj_hom()
        adj_het = self.estimator_adj_het()

        self.model.train()
        self.optimizer_gnn.zero_grad()

        logp = self.model(features, adj_hom, adj_het)

        cla_loss = F.nll_loss(logp[idx_train], labels[idx_train])
        loss = cla_loss
        train_acc = accuracy(logp[idx_train], labels[idx_train])

        loss.backward()
        self.optimizer_gnn.step()

        self.model.eval()
        logp = self.model(features, adj_hom, adj_het)
        test_acc = accuracy(logp[idx_test], labels[idx_test])
        loss_val = F.nll_loss(logp[idx_val], labels[idx_val]).item()
        val_acc = accuracy(logp[idx_val], labels[idx_val])
        self.los.append([epoch, loss_val, val_acc, test_acc])

        return loss_val, train_acc, val_acc, test_acc

    def train_adj(self, epoch, features, labels, idx_train, idx_val, idx_test, adj):

        estimator_adj_hom = self.estimator_adj_hom
        estimator_adj_hom.train()
        self.optimizer_adj_hom.zero_grad()

        estimator_adj_het = self.estimator_adj_het
        estimator_adj_het.train()
        self.optimizer_adj_het.zero_grad()

        loss_fro = torch.norm(estimator_adj_hom() + estimator_adj_het() - adj, p='fro')
        loss_symmetric_hom = torch.norm(estimator_adj_hom() - estimator_adj_hom().t(), p="fro")
        loss_symmetric_het = torch.norm(estimator_adj_het() - estimator_adj_het().t(), p="fro")
        loss_symmetric = loss_symmetric_hom + loss_symmetric_het

        loss_smooth_feat = 0
        if self.args.lambda_:
            loss_smooth_feat_hom = self.feature_smoothing(estimator_adj_hom(), features)
            loss_smooth_feat_het = - self.feature_smoothing(estimator_adj_het(), features)
            loss_smooth_feat = loss_smooth_feat_hom + loss_smooth_feat_het
            loss_smooth_ori = self.feature_smoothing(adj, features)
            print(loss_smooth_feat_hom.item(), loss_smooth_feat_het.item(), loss_smooth_ori.item())


        logp = self.model(features, estimator_adj_hom(), estimator_adj_het())
        gnn_loss = F.nll_loss(logp[idx_train], labels[idx_train])
        loss_diffiential = gnn_loss + self.args.gamma * loss_fro + self.args.phi * loss_symmetric + \
                           self.args.lambda_ * loss_smooth_feat
        loss_val = F.nll_loss(logp[idx_val], labels[idx_val]).item()
        val_acc = accuracy(logp[idx_val], labels[idx_val])
        loss_diffiential.backward()

        self.optimizer_adj_hom.step()
        self.optimizer_adj_het.step()

        if self.args.beta != 0:
            self.optimizer_nuclear_hom.zero_grad()
            self.optimizer_nuclear_hom.step()
            self.optimizer_nuclear_het.zero_grad()
            self.optimizer_nuclear_het.step()

        self.optimizer_l1_hom.zero_grad()
        self.optimizer_l1_hom.step()
        self.optimizer_l1_het.zero_grad()
        self.optimizer_l1_het.step()

        self.model.eval()
        logp = self.model(features, estimator_adj_hom(), estimator_adj_het())

        estimator_adj_hom.estimated_adj.data.copy_(torch.clamp(estimator_adj_hom.estimated_adj.data, min=0, max=1))
        estimator_adj_het.estimated_adj.data.copy_(torch.clamp(estimator_adj_het.estimated_adj.data, min=0, max=1))
        if loss_val <= self.min_loss and self.max_acc <= val_acc:

            self.best_graph_hom = estimator_adj_hom()
            self.best_graph_het = estimator_adj_het()
        else:
            pass

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        L = r_mat_inv @ L @ r_mat_inv
        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat


class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, num_of_nodes, hom=True, symmetric=False, device='cpu'): 
        super(EstimateAdj, self).__init__()

        self.num_of_nodes = num_of_nodes
        self.adj = adj
        self.hom = hom
        self.estimated_adj = nn.Parameter(torch.FloatTensor(self.num_of_nodes, self.num_of_nodes))
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device



    def _init_estimation(self, adj):
        with torch.no_grad():
            if self.hom:
  
                self.estimated_adj.data.copy_(adj)
            else:
                self.estimated_adj.data.copy_(torch.eye(self.num_of_nodes))



    def forward(self):
        return self.estimated_adj * self.adj

    def sample_gumbel(self, shape, eps=1e-20):
        """Sample from Gumbel distribution"""
        U = torch.rand(shape).to(self.device)
        return -torch.log(-torch.log(U + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        """Draw a sample from the Gumbel-Softmax distribution"""
        gumbel_noise = self.sample_gumbel(logits.shape)
        gumbel_logits = (logits + gumbel_noise) / temperature
        return torch.softmax(gumbel_logits, dim=0)

