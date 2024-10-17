# coding=utf-8
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This demo script aim to demonstrate
how to use MCSL algorithm in `castle` package for causal inference.

If you want to plot causal graph, please make sure you have already install
`networkx` package, then like the following import method.

Warnings: This script is used only for demonstration and cannot be directly
          imported.
"""

from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, IIDSimulation
from castle.algorithms import MCSL
import argparse
import torch
import numpy as np
import math
import scipy.stats as st
from itertools import combinations, permutations

def getCorr(data, _x, _y):
    x = data[_x]
    y = data[_y]
    cov = np.mean(x * y) - np.mean(x) * np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    if cov > 0:
        return np.sqrt(cov ** 2 / (var_x * var_y))
    else:
        return -np.sqrt(cov ** 2 / (var_x * var_y))


def indepTest(data, x, y, alpha):  #
    pcc = getCorr(data, x, y)
    zpcc = 0.5 * math.log((1 + pcc) / (1 - pcc))
    A = math.sqrt(data.shape[1] - 3) * math.fabs(zpcc)
    pc_A = (1 - st.norm.cdf(abs(A))) * 2
    B = st.norm.ppf(
        1 - alpha / 2)  # Inverse Cumulative Distribution Function of normal Gaussian (parameter : 1-alpha/2)
    '''
    if A > B:
        return False
    else:
        return True
    '''
    #'''
    if pc_A>alpha:
        return pc_A
    else:
        return 0
    #'''


def oldCI_test(data, max_size, x, y, Z):
    num_nodes = len(data)
    corr = np.zeros([num_nodes, num_nodes, max_size])
    vis = np.zeros([num_nodes, num_nodes, max_size], dtype=bool)
    if len(Z) == 0:
        val = getCorr(data, x, y)
        return val

    def getCorr_cond(x, y, z, k):

        if (vis[x][y][k] == True):
            return corr[x][y][k]
        if (k == len(Z)):
            return getCorr(data, x, y)
        vis[x][y][k] = True
        val_1 = getCorr_cond(x, Z[k], z, k + 1)
        val_2 = getCorr_cond(Z[k], y, Z, k + 1)
        val = getCorr_cond(x, y, Z, k + 1)
        corr[x][y][k] = (val - val_1 * val_2) / (math.sqrt(1 - val_1 * val_1) * math.sqrt(1 - val_2 * val_2))
        # print('({},{},{}) {:.3f}'.format(i,j,k,corr[x][y][k]))
        return corr[x][y][k]

    val = getCorr_cond(x, y, Z, 0)
    return val


def newCI_test(data, x, y, Z, alpha):
    data_x = np.transpose(data[x])  # num_samples, 转置交换矩阵行列
    data_y = np.transpose(data[y])  # num_samples,
    data_Z = np.transpose(data[Z, :])  # num_samples * |Z|
    num_samples = data.shape[1]

    Z_nodes = len(Z)  # length of Z
    if Z_nodes == 0:
        pcc = getCorr(data, x, y)
    else:
        num_samples = len(data_Z)  # number of data samples
        arr_one = (np.ones([num_samples]))
        data_Z = np.insert(data_Z, 0, arr_one, axis=1)  # insert an all-ones column in the left

        wx = np.linalg.lstsq(data_Z, data_x, rcond=None)[
            0]  # wx is the answer of data_Z * X = data_x by using least square method
        wy = np.linalg.lstsq(data_Z, data_y, rcond=None)[
            0]  # wy is the answer of data_Z * X = data_y by using least square method

        rx = data_x - data_Z @ wx  # calc residual error of data_x
        ry = data_y - data_Z @ wy  # calc residual error of data_y

        pcc = num_samples * (np.transpose(rx) @ ry) - np.sum(rx) * np.sum(ry)
        pcc /= math.sqrt(num_samples * (np.transpose(rx) @ rx) - np.sum(rx) * np.sum(rx))
        pcc /= math.sqrt(num_samples * (np.transpose(ry) @ ry) - np.sum(ry) * np.sum(ry))

    zpcc = 0.5 * math.log((1 + pcc) / (1 - pcc))
    A = math.sqrt(num_samples - Z_nodes - 3) * math.fabs(zpcc)
    pc_A = (1 - st.norm.cdf(abs(A))) * 2
    B = st.norm.ppf(
        1 - alpha / 2)  # Inverse Cumulative Distribution Function of normal Gaussian (parameter : 1-alpha/2)
    '''
    if A > B:
        return False
    else:
        return True
    '''
    return pc_A

def prepare_CI_table1(data, order, alpha_0, alpha_1):
    num_nodes = data.shape[1]
    data_matrix = data.T
    CI_table = np.zeros([num_nodes, num_nodes])
    for (x, y) in combinations(range(num_nodes), 2):
        CI_table[x][y] = indepTest(data_matrix, x, y, alpha_0)
        CI_table[y][x] = CI_table[x][y]
        if order == 1:
            if CI_table[x][y] == 0:
                for z in range(num_nodes):
                    if z == x or z == y:
                        continue
                    if newCI_test(data_matrix, x, y, (z,), alpha_1) > alpha_1:
                        CI_table[x][y] = newCI_test(data_matrix, x, y, (z,), alpha_1)
                        CI_table[y][x] = CI_table[x][y]
                        #print(CI_table)
                        break

    # np.savetxt('data1.txt', CI_table, fmt='%f', delimiter=',')
    return CI_table


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
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--x_dims', type=int, default=1,  # changed here
                        help='The number of input dimensions: default 1.')
parser.add_argument('--z_dims', type=int, default=1,
                        help='The number of latent variable dimensions: default the same as variable size.')
parser.add_argument('--lambda_CI', type=float, default=0.5,
                        help='coefficient for CI constraint.')
parser.add_argument('--CI_order', type=int, default=0,
                        help='highest for calculate CI matrix.')
parser.add_argument('--hard_constraint', type=int, default=3,
                        help='use CIC(1, hard constraint) or CIR(0, soft constraint) or (3, no constraint)')
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

n, d = args.data_sample_size, args.data_variable_size
graph_type, degree, sem_type, linear_type = args.graph_type, args.graph_degree, args.graph_sem_type, args.graph_linear_type
x_dims = args.x_dims

#######################################
# mcsl used simulate data
#######################################
# simulate data for mcsl
G, G_true, WG_true = DAG.erdos_renyi(args.dataset, d, degree,
                                      args.seed, weight_range=(0.5, 2.0))
dataset = IIDSimulation(n, W=WG_true, method='nonlinear', sem_type='mlp')
dag, X = dataset.B, dataset.X
CI_table = prepare_CI_table1(X, args.CI_order, args.significance_level_0, args.significance_level_1)
# mcsl learn
mc = MCSL(model_type='nn', num_hidden_layers=8, hidden_dim=4,
                 graph_thresh=0.7, l1_graph_penalty=2e-3, learning_rate=1e-3,
                 max_iter=25, iter_step=100, init_iter=2, h_tol=1e-10,
                 init_rho=1e-4, rho_thresh=1e20, h_thresh=0.25,
                 rho_multiply=10, temperature=0.2, device_type='cpu',
                 device_ids='0', random_seed=args.seed, CI_table=CI_table, args=args)
mc.learn(X, pns_mask=dag)

# plot est_dag and true_dag
GraphDAG(mc.causal_matrix, dag)

# calculate accuracy
met = MetricsDAG(mc.causal_matrix, dag)
print(met.metrics)
