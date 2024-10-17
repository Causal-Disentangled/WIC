# coding=utf-8
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

import numpy as np
import torch
import logging

from castle.common.consts import LOG_FREQUENCY, LOG_FORMAT


logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class ALTrainer(object):

    def __init__(self, n, d, model, lr, init_iter, alpha, beta, rho, rho_thresh,
                 h_thresh, l1_penalty, gamma, early_stopping,
                 early_stopping_thresh, seed, CI_table, args, device=None):
        self.n = n
        self.d = d
        self.model = model
        self.lr = lr
        self.init_iter = init_iter

        self.alpha = alpha
        self.beta = beta  # rho_multiply
        self.rho = rho
        self.rho_thresh = rho_thresh
        self.h_thresh = h_thresh  # 1e-8
        self.l1_penalty = l1_penalty
        self.gamma = gamma
        self.early_stopping = early_stopping
        self.early_stopping_thresh = early_stopping_thresh
        self.seed = seed
        self.CI_table = CI_table
        self.args = args
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr)

    def train(self, x, epochs, update_freq):

        alpha, beta, rho = self.alpha, self.beta, self.rho
        h, h_new = np.inf, np.inf
        prev_w_est, prev_mse = None, np.inf
        for epoch in range(1, epochs + 1):
            logging.info(f'Current epoch: {epoch}==================')
            while rho < self.rho_thresh:
                mse_new, h_new, w_new = self.train_step(x,
                                                        update_freq,
                                                        alpha,
                                                        rho)
                if h_new > self.gamma * h:
                    rho *= self.beta
                else:
                    break
            #logging.info(f'Current        h: {h_new}')

            if self.early_stopping:
                if (mse_new / prev_mse > self.early_stopping_thresh
                        and h_new <= 1e-7):
                    return prev_w_est
                else:
                    prev_w_est = w_new
                    prev_mse = mse_new

            # update rules
            w_est, h = w_new, h_new
            alpha += rho * h_new.detach().cpu()

            if h <= self.h_thresh and epoch > self.init_iter:
                break

        return w_est


    def train_step(self, x, update_freq, alpha, rho):

        def CI_test_Loss(weight, CI_table):
            num_nodes = weight.shape[0]
            CI_table = torch.from_numpy(CI_table)
            mat = CI_table.mul(weight)
            return torch.sum(mat * mat)

        curr_mse, curr_h, w_adj = None, None, None
        for _ in range(update_freq):
            hard_constraint = self.args.hard_constraint
            lambda_CI = 1.0
            torch.manual_seed(self.seed)
            curr_mse, w_adj = self.model(x)
            CI_loss = CI_test_Loss(w_adj.clone().detach(), self.CI_table)

            curr_h = compute_h(w_adj)#PICloss
            if hard_constraint == 1:
                curr_h += lambda_CI * CI_loss * curr_h
            loss = ((0.5 / self.n) * curr_mse
                    + self.l1_penalty * torch.norm(w_adj, p=1)
                    + alpha * curr_h + 0.5 * rho * curr_h * curr_h)#PIR
            if hard_constraint == 0:
                loss += lambda_CI * CI_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if _ % LOG_FREQUENCY == 0:
                logging.info(f'Current loss in step {_}: {loss.detach()}')

        return curr_mse, curr_h, w_adj


def compute_h(w_adj):

    d = w_adj.shape[0]
    h = torch.trace(torch.matrix_exp(w_adj * w_adj)) - d

    return h
