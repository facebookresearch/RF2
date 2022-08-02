# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

def init_emb(tensor):
    n, m = tensor.weight.shape
    W = np.random.uniform(low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)).astype(np.float32)
    tensor.weight.data = torch.tensor(W, requires_grad=True)

def init_mlp(tensor):
    n, m, = tensor.weight.shape

    # This works better
    W = np.random.normal(0, 0.0001, size=(n, m)).astype(np.float32)
    #W = np.random.normal(0, np.sqrt(2 / (m + n)), size=(n, m)).astype(np.float32)
    tensor.weight.data = torch.tensor(W, requires_grad=True)

    if tensor.bias is not None:
        # Bias should be zero to not dominate
        b = np.zeros((1, n)).astype(np.float32)
        tensor.bias.data = torch.tensor(b, requires_grad=True)
