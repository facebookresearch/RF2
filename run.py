# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from fl_libs.arg_parser import get_args
from dlrm.data_utils import load_data, map_tiers
from fl_libs.utils import get_models, set_optimizer
from fl_libs.client import Client
from fl_libs.server import Server

from sklearn.metrics import log_loss, roc_auc_score, recall_score, precision_score, f1_score, average_precision_score
import numpy as np
import pickle
import random
import os
import math


def run_training(args):
    print(args)
    if args.heterofl_channel_multiplier != "1.0":
        # My impl is not compatible with dnn l2 reg
        assert(args.l2_reg_dnn == 0.0)

    # Fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.fl:
        run_fl_training(args)
    else:
        run_centralized_training(args)


def run_centralized_training(args):
    x_train, y_train, x_test, y_test, behavior_feature_list, feature_columns = load_data(args.dataset, args.hist_len, args.target_ctr, args.model == "dien", args.logarithm_input, args.standarize_input, args.emb_dim, args.exclude, args.combiner)

    # Run training
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    model = get_models(args, feature_columns, behavior_feature_list, device)
    optim = set_optimizer(model, args.client_optimizer, args.learning_rate)

    model.compile(optim, 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.num_epochs, verbose=2, validation_split=0.0, lr_scheduler=None, x_test=x_test, y_test=y_test)

    pred_ans = model.predict(x_test, 256)
    print(f"Test loss {round(log_loss(y_test, pred_ans), 4)}")
    print(f"Test AUC {round(roc_auc_score(y_test, pred_ans), 4)}")
    print(f"Test ap {round(average_precision_score(y_test, pred_ans), 4)}", flush=True)


def run_fl_training(args):
    world_size = args.ngpus
    print(f"{world_size} GPUs available")

    x_train, y_train, x_test, y_test, behavior_feature_list, feature_columns = load_data(args.dataset, args.hist_len, args.target_ctr, args.model == "dien", args.logarithm_input, args.standarize_input, args.emb_dim, args.exclude, args.combiner)

    # Generate clients
    clients = generate_clients(x_train, y_train, x_test, y_test, args.min_sample_size, args.data_split, args.num_clients, args=args)

    map_tiers(args.dataset, clients, args.tier_selection, args.seed, args.comm_prune_rate, args.comm_quantize_bits, args.comm_quantize_separate_sign, args.heterofl_channel_multiplier)

    train_clients = list(filter(lambda x: len(x.y_train) > 0, clients))
    if args.exclude_low:
        train_clients = list(filter(lambda x: x.device_tier > 0, train_clients))

    print(f"Total clients with tier 0 {len(list(filter(lambda x: x.device_tier == 0, clients)))}")
    print(f"Total clients with tier 1 {len(list(filter(lambda x: x.device_tier == 1, clients)))}")
    print(f"Total clients with tier 2 {len(list(filter(lambda x: x.device_tier == 2, clients)))}")

    print(f"Train clients with tier 0 {len(list(filter(lambda x: x.device_tier == 0, train_clients)))}")
    print(f"Train clients with tier 1 {len(list(filter(lambda x: x.device_tier == 1, train_clients)))}")
    print(f"Train clients with tier 2 {len(list(filter(lambda x: x.device_tier == 2, train_clients)))}")

    test_clients = list(filter(lambda x: len(x.y_test) > 0, clients))
    print(f"Test clients with tier 0 {len(list(filter(lambda x: x.device_tier == 0, test_clients)))}")
    print(f"Test clients with tier 1 {len(list(filter(lambda x: x.device_tier == 1, test_clients)))}")
    print(f"Test clients with tier 2 {len(list(filter(lambda x: x.device_tier == 2, test_clients)))}")

    for c in test_clients:
        features = [k for k in c.x_test]
        break

    total_train_cids = [(c.cid, c.device_tier) for c in train_clients]
    # Organize test data per tiers
    test_data = {tier:
            {"x_test": {k: np.concatenate([c.x_test[k] for c in list(filter(lambda x: x.device_tier == tier, test_clients))], axis=0) for k in features},
                        "y_test": np.concatenate([c.y_test for c in list(filter(lambda x: x.device_tier == tier, test_clients))], axis=0)}
                for tier in range(3)}

    #print([(tier, len(v["y_test"])) for tier, v in test_data.items()])

    # We use spawn a trainer for each GPU. (Local) rank 0 is a trainer + server.
    mp.set_start_method("spawn")
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=init_trainers, args=(args, rank, world_size, run_training, train_clients[rank::world_size], test_data if rank == 0 else None, total_train_cids, feature_columns, behavior_feature_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def init_trainers(args, rank, world_size, fn, train_clients, test_data, total_train_cids, feature_columns, behavior_feature_list, backend='gloo'):
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = str(args.port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    run_fl_per_rank(args, rank, world_size, train_clients, test_data, total_train_cids, feature_columns, behavior_feature_list)


def run_fl_per_rank(args, rank, world_size, train_clients, test_data, total_train_cids, feature_columns, behavior_feature_list):
    print(f"Rank {rank}", flush=True)
    #print(torch.cuda.is_available(), torch.cuda.current_device(), torch.cuda.device_count(), flush=True)
    assert(torch.cuda.is_available())

    # Fix seed (TODO: Do we need to do this per rank?)
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    torch.manual_seed(123 + args.seed)

    device = f"cuda:{rank}"

    model = get_models(args, feature_columns, behavior_feature_list, device)
    model.compile(loss='binary_crossentropy',
                  metrics=['binary_crossentropy'])

    # Generate server
    server = Server(rank, world_size, train_clients, test_data, model, device, args.server_optimizer, args.server_lr, total_train_cids, args.metrics, args.seed, "-" in args.heterofl_channel_multiplier, args.overselect)
    server.init_clients(model, device, args.client_optimizer)

    for epoch in range(args.num_global_epochs):
        # Shuffle client order before each global epoch
        print("Global epoch ", epoch)
        random.shuffle(server.total_train_cids)
        server.train(args.clients_per_round,
                args.num_epochs,
                args.batch_size, args.aggr_method, args.test_freq, args.learning_rate, "sequential")

        if rank == 0:
            print(f"Global epoch {epoch} done")
            server.test()

    if rank == 0:
        server.test(save_result=False)
        print("FL training done", flush=True)


def generate_clients(x_train, y_train, x_test, y_test, min_sample_size, data_split, num_clients, args=None):
    if num_clients == 0:
        assert(data_split == "niid")

    uid_key = "s0"

    # Data per clients
    data_per_clients = {}

    if data_split == "niid":
        for i, uid in enumerate(x_train[uid_key]):
            if uid not in data_per_clients:
                data_per_clients[uid] = {
                        "x_train": {k: [] for k in x_train},
                        "y_train": [],
                        "x_test": {k: [] for k in x_train},
                        "y_test": [],
                }
            for k, v in data_per_clients[uid]["x_train"].items():
                v.append(x_train[k][i])
            data_per_clients[uid]["y_train"].append(y_train[i])
    elif data_split == "iid":
        shuffled_idx = list(range(len(y_train)))
        random.shuffle(shuffled_idx)
        train_data_per_user = math.ceil(len(shuffled_idx) / num_clients)

        test_shuffled_idx = list(range(len(y_test)))
        random.shuffle(test_shuffled_idx)
        test_data_per_user = math.ceil(len(test_shuffled_idx) / num_clients)
        print("Train data per user", train_data_per_user)
        for uid in range(num_clients):
            data_per_clients[uid] = {
                    "x_train": {k: [x_train[k][i]
                            for i in shuffled_idx[
                                uid * train_data_per_user:(uid+1) * train_data_per_user
                            ]]
                        for k in x_train},
                "y_train": [y_train[i] for i in shuffled_idx[
                                uid * train_data_per_user:(uid+1) * train_data_per_user
                            ]],
                "x_test": {k: [x_test[k][i]
                            for i in test_shuffled_idx[
                                uid * test_data_per_user:(uid+1) * test_data_per_user
                            ]]
                        for k in x_test},
                "y_test": [y_test[i] for i in test_shuffled_idx[
                                uid * test_data_per_user:(uid+1) * test_data_per_user
                            ]],
            }
    else:
        raise AssertionError()

    # If we restrict the number of clients, we select by the number of training samples.
    if num_clients > 0:
        data_per_clients = {k:v for k, v in sorted(data_per_clients.items(), key=lambda x: len(x[1]["y_train"]), reverse=True)[:num_clients]}
    else:
        data_per_clients = {k:v for k, v in sorted(data_per_clients.items(), key=lambda x: len(x[1]["y_train"]), reverse=True)}

    # Remove clients with samples less than min_sample_size
    data_per_clients = dict(filter(lambda x: len(x[1]["y_train"]) >= min_sample_size, data_per_clients.items()))

    tot_len = len(y_train)
    included_len = sum([len(data_per_clients[k]["y_train"]) for k in data_per_clients])
    print("Total train data: ", tot_len, " Included train data, ", included_len, f" ({included_len/tot_len * 100}%)", len(data_per_clients), " train clients", flush=True)

    # Add the test samples to each client
    test_users = set()
    unseen_test_users = set()
    if data_split == "niid":
        for i, uid in enumerate(x_test[uid_key]):
            test_users.add(uid)
            if uid not in data_per_clients:
                unseen_test_users.add(uid)
                data_per_clients[uid] = {
                        "x_train": {k: [] for k in x_test},
                        "y_train": [],
                        "x_test": {k: [] for k in x_test},
                        "y_test": [],
                }
            for k, v in data_per_clients[uid]["x_test"].items():
                v.append(x_test[k][i])
            data_per_clients[uid]["y_test"].append(y_test[i])
    elif data_split == "iid":
        # Must be handled above
        pass
    else:
        raise AssertionError()

    print("Test data: ", len(y_test), " Across ", len(test_users), " users, ", len(unseen_test_users), " of them unseen users")

    clients = [Client(uid, {k: np.array(v) for k, v in data_per_clients[uid]["x_train"].items()}, np.array(data_per_clients[uid]["y_train"]),
        {k: np.array(v) for k, v in data_per_clients[uid]["x_test"].items()}, np.array(data_per_clients[uid]["y_test"]), args=args) for i, uid in enumerate(data_per_clients)]

    print(len(clients), " clients", flush=True)

    return clients


if __name__ == "__main__":
    args = get_args()
    run_training(args)
