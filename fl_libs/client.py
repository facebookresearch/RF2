# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import time

class Client:
    def __init__(self, cid, x_train, y_train, x_test, y_test, device_tier=0, comm_prune_rate=0.0, comm_quantize_bits=0, comm_quantize_separate_sign=False, channel_multiplier=1.0, args=None):
        self.cid = cid
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.device_tier = device_tier
        self.args = args # For debugging
        
        # Optimization params
        self.comm_prune_rate = comm_prune_rate
        self.comm_quantize_bits = comm_quantize_bits
        self.comm_quantize_separate_sign = comm_quantize_separate_sign

        self.channel_multiplier = channel_multiplier

    def init_client(self, model, device, optimizer):
        self.model = model
        self.device = device
        self.optimizer = optimizer

    def train(self, epochs, bs, lr):
        # Init optimizer
        if self.optimizer == "sgd":
            optim = torch.optim.SGD(self.model.parameters(), lr=lr)
        elif self.optimizer == "adagrad":
            optim = torch.optim.Adagrad(self.model.parameters(), lr=lr)
        elif self.optimizer == "adam":
            optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.optim = optim

        # For heteroFL
        self.model.set_channel_multiplier(self.channel_multiplier)

        # To calculate each gradient.
        # Calculating and sending gradients instead of
        # the raw model value for future extensibility.
        grad = {}
        for name, param in self.model.state_dict().items():
            grad[name] = param.data.clone().detach()

        # Run training locally
        if bs == 0: # Full-batch training
            bs = len(self.y_train)
        self.model.fit(self.x_train, self.y_train, batch_size=bs, epochs=epochs, verbose=0, validation_split=0.0)

        # Calculate gradient
        for name, param in self.model.state_dict().items():
            grad[name] -= param.data.clone().detach()

        for name, param in self.model.named_parameters():
            # Prune gradient randomly
            if self.comm_prune_rate > 0.0:
                mask = torch.cuda.FloatTensor(grad[name].shape).uniform_() > self.comm_prune_rate
                grad[name] *= mask

            # Stochastic quantization
            if self.comm_quantize_bits >= 1:
                if self.comm_quantize_separate_sign:
                    min_t = 0
                    max_t = torch.max(torch.abs(grad[name]))

                else:
                    min_t = torch.min(grad[name])
                    max_t = torch.max(grad[name])

                if min_t == max_t:
                    continue

                if self.comm_quantize_separate_sign:
                    normalized_t = (torch.abs(grad[name]) - min_t) / (max_t - min_t)
                    sign_t = ((grad[name] > 0).float() - 0.5) * 2

                else:
                    normalized_t = (grad[name] - min_t) / (max_t - min_t)

                # 2 ** b representative points, 2 ** b - 1 intervals
                intervals = 2 ** self.comm_quantize_bits - 1
                normalized_t *= intervals

                floor_t = torch.floor(normalized_t)

                comp = torch.rand_like(floor_t).to(self.device)

                delta = normalized_t - floor_t
                floor_t += (delta > comp).float()

                grad[name] = floor_t / intervals * (max_t - min_t) + min_t

                if self.comm_quantize_separate_sign:
                    grad[name] *= sign_t

        num_train_samples = self.y_train.shape[0]

        return num_train_samples, grad
