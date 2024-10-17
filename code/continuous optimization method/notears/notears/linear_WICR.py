import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
import argparse
import networkx as nx
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import numpy as np
import sys
import os
# np.set_printoptions(threshold=np.inf)
import math
from itertools import combinations, permutations
import scipy.stats as st
from utils import *


def notears_linear(X, lambda1, loss_type, max_iter=300, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def matrix_poly(matrix, d):
        x = torch.eye(d).double() + torch.div(torch.from_numpy(matrix), d)
        return torch.matrix_power(x, d)

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        #expm_A = matrix_poly(W * W, d)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        #G_h = G_h.numpy()
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        CI_loss = CI_test_Loss(torch.tensor(W), CI_table)
        CI_loss = CI_loss.numpy()
        if args.hard_constraint == True:
            h += args.lambda_CI * CI_loss * beta
        else:
            loss += args.lambda_CI * CI_loss
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h, beta = np.zeros(2 * d * d), 1.0, 0.0, np.inf, 1.0  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
                beta *= 0.2
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est

def load_data(args, batch_size=1000, debug = False):
    #  # configurations
    n, d = args.data_sample_size, args.data_variable_size
    graph_type, degree, sem_type, linear_type = args.graph_type, args.graph_degree, args.graph_sem_type, args.graph_linear_type
    x_dims = args.x_dims

    if args.data_type == 'synthetic':
        # generate data
        G, G_true, WG_true = simulate_random_dag(args.dataset, d, degree, graph_type)
        X = simulate_sem(G, n, x_dims, sem_type, linear_type)

    elif args.data_type == 'discrete':
        # get benchmark discrete data
        if args.data_filename.endswith('.pkl'):
            with open(os.path.join(args.data_dir, args.data_filename), 'rb') as handle:
                X = pickle.load(handle)
        else:
            all_data, graph = read_BNrep(args)
            G = nx.DiGraph(graph)
            X = all_data['1000']['1']
    #print(X)

    feat_train = torch.FloatTensor(X)
    feat_valid = torch.FloatTensor(X)
    feat_test = torch.FloatTensor(X)

    # reconstruct itself
    train_data = TensorDataset(feat_train, feat_train)
    valid_data = TensorDataset(feat_valid, feat_train)
    test_data = TensorDataset(feat_test, feat_train)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, G, X, G_true, WG_true


if __name__ == '__main__':
    import utils
    utils.set_random_seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='synthetic',
                        choices=['synthetic', 'discrete', 'real'],
                        help='choosing which experiment to do.')
    parser.add_argument('--data_filename', type=str, default='Alarm',
                        help='data file name containing the discrete files.')
    parser.add_argument('--dataset', type=str, default='alarm',
                        help='Use which dataset.')
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='data file name containing the discrete files.')
    parser.add_argument('--data_sample_size', type=int, default=200,
                        help='the number of samples of data')
    parser.add_argument('--data_variable_size', type=int, default=10,
                        help='the number of variables in synthetic generated data')
    parser.add_argument('--graph_type', type=str, default='erdos-renyi',
                        help='the type of DAG graph by generation method')
    parser.add_argument('--graph_degree', type=int, default=2,
                        help='the number of degree in generated DAG graph')
    parser.add_argument('--graph_sem_type', type=str, default='linear-uniform',
                        help='the structure equation model (SEM) parameter type')
    parser.add_argument('--graph_linear_type', type=str, default='nonlinear_2',
                        help='the synthetic data type: linear -> linear SEM, nonlinear_1 -> x=Acos(x+1)+z, nonlinear_2 -> x=2sin(A(x+0.5))+A(x+0.5)+z')
    parser.add_argument('--batch-size', type=int, default=100,
                        # note: should be divisible by sample size, otherwise throw an error
                        help='Number of samples per batch.')
    parser.add_argument('--seed', type=int, default=8, help='Random seed.')
    parser.add_argument('--degree',
                        type=int,
                        default=4,
                        help="Degree of graph.")
    parser.add_argument('--noise_type',
                        type=str,
                        default='uniform',
                        help="Type of noise ['uniform', 'gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'].")
    parser.add_argument('--x_dims', type=int, default=1,  # changed here
                        help='The number of input dimensions: default 1.')
    parser.add_argument('--z_dims', type=int, default=1,
                        help='The number of latent variable dimensions: default the same as variable size.')
    parser.add_argument('--lambda_CI', type=float, default=0.5,
                        help='coefficient for CI constraint.')
    parser.add_argument('--CI_order', type=int, default=0,
                        help='highest for calculate CI matrix.')
    parser.add_argument('--hard_constraint', type=int, default=1,
                        help='use CIC(1, hard constraint) or CIR(0, soft constraint)')
    parser.add_argument('--significance_level_0', type=float, default=0.01,
                        help='significance level for CI in order 0.')
    parser.add_argument('--significance_level_1', type=float, default=0.1,
                        help='significance level for CI in order 1.')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='Disables CUDA training.')
    parser.add_argument('--prior', action='store_true', default=False,
                        help='Whether to use sparsity prior.')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    file = r'skeleton_{}.csv'.format(args.dataset)
    with open(file, encoding='utf-8') as f:
        skeleton_data = np.loadtxt(file, delimiter=',')
    args.data_variable_size = skeleton_data.shape[0]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    B_scale = 1.0
    dataset = SyntheticDataset(args.data_sample_size, args.data_variable_size, args.dataset, args.graph_type,
                               args.degree, args.noise_type, B_scale, args.seed)
    dataset.X -= np.mean(dataset.X, axis=0)
    train_loader, valid_loader, test_loader, ground_truth_G, X1, G_true, WG_true = load_data(args, args.batch_size)
    all_data = []
    for batch_idx, (data, relations) in enumerate(train_loader):
        all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    all_data -= np.mean(all_data, axis=0)
    CI_table = prepare_CI_table1(dataset.X, args.CI_order, args.significance_level_0, args.significance_level_1)

    np.savetxt('WG_true.csv', WG_true, delimiter=',')
    np.savetxt('X.csv', dataset.X, delimiter=',')

    W_est = notears_linear(dataset.X, lambda1=0.1, loss_type='l2')
    assert utils.is_dag(W_est)
    np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = utils.count_accuracy(G_true, W_est != 0)
    print(acc)

