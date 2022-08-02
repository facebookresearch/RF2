# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import DNN

from deepctr_torch.inputs import DenseFeat, SparseFeat, VarLenSparseFeat

class DLRM(BaseModel):
    def __init__(self,
                 dnn_feature_columns,
                 bottom_mlp_hidden_units=(16,),
                 top_mlp_hidden_units=(1024, 512, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):

        super(DLRM, self).__init__([], dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)

        self.dnn_feature_columns = dnn_feature_columns

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if len(dnn_feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if len(dnn_feature_columns) else []
        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if len(dnn_feature_columns) else []

        # Assume uniform emb_dim
        emb_dim = sparse_feature_columns[0].embedding_dim

        # Bottom MLP
        if len(dense_feature_columns) > 0:
            self.bottom_mlp = DNN(len(dense_feature_columns), list(bottom_mlp_hidden_units) + [emb_dim], l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)

        # Top MLP
        self.top_mlp = DNN(len(dense_feature_columns) * emb_dim + (len(dnn_feature_columns) * (len(dnn_feature_columns) - 1)) // 2, top_mlp_hidden_units, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(
            top_mlp_hidden_units[-1], 1, bias=False).to(device)
        if l2_reg_dnn > 0:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.bottom_mlp.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.top_mlp.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):
        # Emb lookup + pooling
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)

        if len(dense_value_list) > 0:
            dense_x = torch.cat(dense_value_list, dim=1)
            #print(dense_x.shape)

            # Bottom MLP
            bot_x = self.bottom_mlp(dense_x)
            #print(dense_x.shape)
        else:
            bot_x = torch.Tensor([]).to(sparse_embedding_list[0].device)

        # Feature interaction
        z = self.interact_features(bot_x, [t.reshape([-1, t.shape[-1]]) for t in sparse_embedding_list])

        # Top MLP
        dnn_output = self.top_mlp(z)
        logit = self.dnn_linear(dnn_output)

        y_pred = self.out(logit)

        return y_pred, 0, 0

    def interact_features(self, x, ly):
        # Code directly borrowed from DLRM
        #print(len(ly), ly[0].shape)

        #if x.shape[0] == 0:
        (batch_size, d) = ly[0].shape
        #else:
        #    (batch_size, d) = x.shape

        T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
        #print(T.shape)
        # perform a dot product
        Z = torch.bmm(T, torch.transpose(T, 1, 2))
        # append dense feature with the interactions (into a row vector)
        _, ni, nj = Z.shape
        li = torch.tensor([i for i in range(ni) for j in range(i)])
        lj = torch.tensor([j for i in range(nj) for j in range(i)])
        Zflat = Z[:, li, lj]
        # concatenate dense features and interactions
        #print(x.shape, Zflat.shape)
        R = torch.cat([x] + [Zflat], dim=1)

        return R

    def set_channel_multiplier(self, channel_multiplier):
        self.top_mlp.channel_multiplier = channel_multiplier

