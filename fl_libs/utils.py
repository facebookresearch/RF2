from deepctr_torch.models.din import DIN
from deepctr_torch.models.dlrm import DLRM

import torch

def get_models(args, feature_columns, behavior_feature_list, device):
    dnn_hidden_units = [int(x) for x in args.dnn_hidden_units.split("-")]
    att_hidden_size = [int(x) for x in args.att_hidden_units.split("-")]

    # Models
    if args.model == "din":
        model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=args.att_weight_normalization, dnn_hidden_units=dnn_hidden_units, att_hidden_size=att_hidden_size, l2_reg_embedding=args.l2_reg_emb, l2_reg_dnn=args.l2_reg_dnn, dice_norm_type=args.dice_norm_type, att_activation=args.att_activation)
    elif args.model == "dlrm":
        model = DLRM(feature_columns, device=device, bottom_mlp_hidden_units=[16], top_mlp_hidden_units=dnn_hidden_units, l2_reg_embedding=args.l2_reg_emb, l2_reg_dnn=args.l2_reg_dnn)

    print("Get model done")
    return model


def set_optimizer(model, optimizer, lr):
    if optimizer == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer == "adagrad":
        optim = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "amsgrad":
        optim = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)

    return optim
