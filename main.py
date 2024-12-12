import argparse
import torch
import time
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp

from utils import load_data, accuracy, normalize_adjacency
from model import FAGCN_wodgl, SplitGNN
torch.autograd.set_detect_anomaly(True)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=10, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='cornell',
                        choices=['cora', 'citeseer', 'pubmed', 'chameleon',
                                 'cornell', 'texas', 'squirrel', 'wisconsin'], help='data')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--eps', type=float, default=0.5,
                        help='Fixed scalar or learnable weight.')
    parser.add_argument('--layer_num', type=int, default=3, help='Number of layers')
    parser.add_argument('--lr', type=float, default=0.03289666123287841, help='Initial learning rate.')
    parser.add_argument('--lr_adj', type=float, default=0.002018508629298277, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=3.6953610167628876e-05,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--epochs', type=int,  default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Ratio of training set')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Ratio of testing set')
    parser.add_argument('--pre_splitted', type=bool, default=False,
                        help='For homo graph, True: semi-supervise, False: full-supervise + random split. For heter graph, False: random split')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--train_gnn_steps', type=int, default=1, help='steps for train_gcn optimization')
    parser.add_argument('--train_adj_steps', type=int, default=1, help='steps for train_adj optimization')

    parser.add_argument('--alpha', type=float, default=0.0286118382967511, help='weight of l1 norm')
    parser.add_argument('--beta', type=float, default=1, help='weight of nuclear norm')
    parser.add_argument('--gamma', type=float, default=0.0005238078763948325, help='weight of loss fro')
    parser.add_argument('--phi', type=float, default=1.67580132546942e-05, help='weight of symmetric loss')
    parser.add_argument('--lambda_', type=float, default=0.0017555958671075674, help='weight of feature smoothing')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:1" if args.cuda else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    adj, features, labels, num_features, num_classes, idx_train, idx_val, idx_test = load_data(args.dataset, args.seed, args.device,
                                                                              args.train_ratio, args.val_ratio, args.test_ratio,
                                                                              args.pre_splitted)

    model = FAGCN_wodgl(labels, adj, num_features, args.hidden, num_classes, args.dropout, args.eps, args.layer_num)
    model = model.to(args.device)

    splitgnn = SplitGNN(model, args)
    splitgnn.fit(features, adj, labels, idx_train, idx_val, idx_test)
