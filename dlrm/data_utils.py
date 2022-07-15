# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: generate inputs and targets for the DLRM benchmark

# Copyright (c) Meta Platforms, Inc. All Rights Reserved
#
# Part of the source code was taken from the DLRM project and was
# modified for RF^2 project.

from __future__ import absolute_import, division, print_function, unicode_literals

from deepctr_torch.inputs import DenseFeat, SparseFeat, VarLenSparseFeat

import sys
import pickle
from os import path

import numpy as np
import datetime

import random
import math

clk_thres = 5 # Hardcoded for MovieLens

def map_tiers(dataset, clients, tier_selection, seed, comm_prune_rate, comm_quantize_bits, q_separate_sign, heterofl_channel_multiplier):
    if "random" in tier_selection:
        tier_map_func = lambda i, uid: i % 3
    elif "dirichlet" in tier_selection:
        # In the form of dirichlet-s1-0.5
        dirichlet_feat = tier_selection.split("-")[1]
        alpha = float(tier_selection.split("-")[2])

        f = f"./processed/{dataset}_tier_mapping_dirichlet_{dirichlet_feat}_{alpha}_{seed}"
        if dataset == "movielens-20":
            f += f"_{clk_thres}star"
        f += ".pkl"
        if path.exists(f):
            print(f"Reading existing tier mapping.. {f}")
            with open(f, "rb") as handle:
                tier_mapping = pickle.load(handle)
        else:
            print(f"Generating tier mapping.. {f}")

            item_uids = {-1: []}
            
            # Generate client-item mapping
            data = {}
            for c in clients:
                data[c.cid] = []
                for i, y in enumerate(c.y_train):
                    if y == 1:
                        data[c.cid].append(c.x_train[dirichlet_feat][i])
                for i, y in enumerate(c.y_test):
                    if y == 1:
                        data[c.cid].append(c.x_test[dirichlet_feat][i])

            tiers = {0:[], 1:[], 2:[]}
            no_click_users = []

            for i, (uid, items) in enumerate(data.items()):
                if len(items) == 0:
                    item_uids[-1].append(uid)
                else:
                    for item in items:
                        if item not in item_uids:
                            item_uids[item] = []
                        item_uids[item].append(uid)

            item_cnt = [(len(v), k) for k, v in item_uids.items()]
            item_cnt.sort()

            seen_uids = set()

            for i, (_, item) in enumerate(item_cnt[::-1]):
                dists = [float("nan")] * 3
                while math.isnan(dists[0]) or math.isnan(dists[1]) or math.isnan(dists[2]):
                    dists = np.random.dirichlet([alpha] * 3)
                dists.sort()
                dists = dists[::-1]
                
                cur_tier_size = [(len(v), k) for k, v in tiers.items()]
                cur_tier_size.sort()

                uids_to_map = []
                for uid in item_uids[item]:
                    if uid not in seen_uids:
                        uids_to_map.append(uid)
                        seen_uids.add(uid)
                if len(uids_to_map) == 0:
                    continue
                random.shuffle(uids_to_map)
                dists = [d * len(uids_to_map) for d in dists]

                # Currently smallest tier
                try:
                    tiers[cur_tier_size[0][1]] += uids_to_map[:int(dists[0])]
                    tiers[cur_tier_size[1][1]] += uids_to_map[int(dists[0]):int(dists[0]+dists[1])]
                    tiers[cur_tier_size[2][1]] += uids_to_map[int(dists[0]+dists[1]):]
                except Exception as e:
                    print(uids_to_map, dists, e)

            print(len(tiers[0]))
            print(len(tiers[1]))
            print(len(tiers[2]))

            tier_mapping = {}
            for tier, uids in tiers.items():
                for uid in uids:
                    tier_mapping[uid] = tier

            with open(f, "wb") as handle:
                pickle.dump(tier_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saving {f} done")

        tier_map_func = lambda i, uid: tier_mapping[uid]

    # Optimization 1: random pruning of gradient when communicating.
    if "-" not in comm_prune_rate:
        cpr_rate_dict = {tier: float(comm_prune_rate) for tier in [0, 1, 2]}
    else:
        cpr_rate_dict = {tier: float(val) for tier, val in enumerate(comm_prune_rate.split("-"))}

    # Optimization 2: gradient quantization.
    if "-" not in comm_quantize_bits:
        q_dict = {tier: int(comm_quantize_bits) for tier in [0, 1, 2]}
    else:
        q_dict = {tier: int(val) for tier, val in enumerate(comm_quantize_bits.split("-"))}

    # Optimization 3: channel multiplier
    if "-" not in heterofl_channel_multiplier:
        multiplier_dict = {tier: float(heterofl_channel_multiplier) for tier in [0, 1, 2]}
    else:
        multiplier_dict = {tier: float(val) for tier, val in enumerate(heterofl_channel_multiplier.split("-"))}

    for i, c in enumerate(clients):
        c.device_tier = tier_map_func(i, c.cid)
        c.comm_prune_rate = cpr_rate_dict[c.device_tier]
        c.comm_quantize_bits = q_dict[c.device_tier]
        c.channel_multiplier = multiplier_dict[c.device_tier]
        c.comm_quantize_separate_sign = q_separate_sign

def preprocess_input_taobao(X_int, X_cat, y, dense_features, sparse_features, history_len, seq_maxlen):
    x = {}

    for i, name in enumerate(sparse_features + dense_features):
        print(i, name)
        if i >= len(sparse_features):
            # Dense
            print("Dense")
            x[name] = np.array([x[i - len(sparse_features)] for x in X_int])

        elif isinstance(X_cat[0][i], list):
            # Multi-hot sparse (hist)
            print("Multi-hot sparse")
            x[name] = np.array([x[i][-history_len:] + [0] * (seq_maxlen - len(x[i][-history_len:])) for x in X_cat])
        else:
            # One-hot sparse
            print("One-hot sparse")
            x[name] = np.array([x[i] for x in X_cat])

    y = np.array(y)

    return x, y


def preprocess_input_movielens(X_cat, y, sparse_features, history_len, seq_maxlen):
    x = {}

    for i, name in enumerate(sparse_features):
        print(i, name)
        if isinstance(X_cat[0][i], list):
            # Multi-hot sparse (hist)
            print("Multi-hot sparse")
            x[name] = np.array([x[i][-history_len:] + [0] * (seq_maxlen - len(x[i][-history_len:])) for x in X_cat])
        else:
            # One-hot sparse
            print("One-hot sparse")
            x[name] = np.array([x[i] for x in X_cat])

    y = np.array(y)

    return x, y


def load_data(dataset, hist_len, target_ctr, use_neg_hist, logarithm_input, standarize_input, emb_dim, exclude, combiner):
    if target_ctr > 0:
        ctr_str = f"_{target_ctr}"
    else:
        ctr_str = ""

    processed_file = f"./processed/{dataset}_processed_input_{hist_len}{ctr_str}"
    if dataset == "movielens-20":
        processed_file += f"_{clk_thres}star"
    if use_neg_hist:
        processed_file += f"_w_neg"

    processed_file += ".pkl"

    if path.exists(processed_file):
        print("Reading processed input..")
        with open(processed_file, 'rb') as handle:
            data = pickle.load(handle)
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]
        behavior_feature_list = data["behavior_feature_list"]
        feature_columns = data["feature_columns"]
    else:
        print("Reading FB-DLRM-compatible input..")
        
        fname = "processed"
        if dataset == "movielens-20":
            fname += f"_{clk_thres}star"
        if use_neg_hist:
            fname += "_neg"
        fpath = f"/checkpoint/kwmaeng/{dataset}/{fname}.pkl"
        if not path.exists(fpath):
            print("FB-DLRM-compatible input generation")
            if dataset == "movielens-20":
                getMovieLensData("/checkpoint/kwmaeng/movielens-20/", fpath, clk_thres=clk_thres)
            elif dataset == "taobao":
                getTaobaoAdsData("/checkpoint/kwmaeng/taobao/", fpath, use_neg=use_neg_hist)
            else:
                raise AssertionError("Unsupported dataset")

        with open(fpath, 'rb') as handle:
            data = pickle.load(handle)

        if dataset == "taobao":
            behavior_feature_list = ['s9', 's10', 's13']
            dense_features = ['d' + str(i) for i in range(len(data["X_int"][0]))]
            sparse_features = []
            for i, x in enumerate(data["X_cat"][0]):
                if not isinstance(x, list):
                    sparse_features.append(f"s{i}")
            seq_maxlen = max(np.array([min(len(x[-1]), hist_len) for x in data["X_cat"]]))
            varlen_sparse_features = [f'hist_{x}' for x in behavior_feature_list]

        elif dataset == "movielens-20":
            behavior_feature_list = ['s1'] # Only movie id history for now
            sparse_features = ['s0', 's1', 's2']
            seq_maxlen = hist_len
            dense_features = []
            varlen_sparse_features = [f'hist_{x}' for x in behavior_feature_list] + ['s2']

        else:
            raise AssertionError("Unsupported dataset")

        sparse_features += [f'hist_{x}' for x in behavior_feature_list]
        if use_neg_hist:
            sparse_features += [f"neg_hist_{x}" for x in behavior_feature_list]

        print(dense_features, sparse_features, data["counts"], seq_maxlen)

        feature_columns = [
                VarLenSparseFeat(SparseFeat(feat, data["counts"][i] + 1, embedding_dim=16), seq_maxlen, combiner=combiner, length_name=f"seq_length_{feat}")
                if feat in varlen_sparse_features else
                SparseFeat(feat, data["counts"][i], embedding_dim=16)
                for i, feat in enumerate(sparse_features)]
        feature_columns += [DenseFeat(feat, 1) for feat in dense_features]

        if dataset == "taobao":
            print({k: len(v) for k, v in data.items()})
            data_train = {key: data[key][:21929918] for key in ["X_int", "X_cat", "y"]}
            data_test = {key: data[key][21929918:] for key in ["X_int", "X_cat", "y"]}

            curCtr = sum(data['y']) / len(data['y'])
            print(f"Baseline CTR of the dataset {curCtr}")
            if target_ctr == -1:
                pass
            elif target_ctr > curCtr:
                # Subsample zeros
                subsample_rate = (curCtr * (1 - target_ctr)) / (target_ctr * (1 - curCtr))
                train_zero_idx = list(filter(lambda x: data_train["y"][x] == 0, range(len(data_train["y"]))))
                test_zero_idx = list(filter(lambda x: data_test["y"][x] == 0, range(len(data_test["y"]))))
                train_remove_idx = random.sample(train_zero_idx, int(len(train_zero_idx) * (1 - subsample_rate)))
                test_remove_idx = random.sample(test_zero_idx, int(len(test_zero_idx) * (1 - subsample_rate)))
                train_idx = list(set(range(len(data_train["y"]))) - set(train_remove_idx))
                train_idx.sort()
                test_idx = list(set(range(len(data_test["y"]))) - set(test_remove_idx))
                test_idx.sort()

                data_train = {k: [v[i] for i in train_idx] for k, v in data_train.items()}
                data_test = {k: [v[i] for i in test_idx] for k, v in data_test.items()}
                print(f"After subsampling: {len(data_train['y'])} {sum(data_train['y']) / len(data_train['y'])}")
                print(f"After subsampling: {len(data_test['y'])} {sum(data_test['y']) / len(data_test['y'])}")
            else:
                assert(False)

            x_train, y_train = preprocess_input_taobao(data_train["X_int"], data_train["X_cat"], data_train["y"], dense_features, sparse_features, hist_len, seq_maxlen)
            x_test, y_test = preprocess_input_taobao(data_test["X_int"], data_test["X_cat"], data_test["y"], dense_features, sparse_features, hist_len, seq_maxlen)
            for feat in behavior_feature_list:
                x_train[f"seq_length_hist_{feat}"] = np.array([min(len(x[-1]), hist_len) for x in data_train["X_cat"]])
                x_test[f"seq_length_hist_{feat}"] = np.array([min(len(x[-1]), hist_len) for x in data_test["X_cat"]])

        elif dataset == "movielens-20":
            curCtr = (sum(data['y_train']) + sum(data['y_test'])) / (len(data['y_train']) + len(data['y_test']))
            print(f"Baseline CTR of the dataset {curCtr}")

            if target_ctr == -1:
                pass
            elif target_ctr > curCtr:
                # Subsample zeros
                raise AssertionError("Subsampling not implemented!")
            else:
                assert(False)

            x_train, y_train = preprocess_input_movielens(data["x_train"], data["y_train"], sparse_features, hist_len, seq_maxlen)
            x_test, y_test = preprocess_input_movielens(data["x_test"], data["y_test"], sparse_features, hist_len, seq_maxlen)
            x_train["seq_length_s2"] = np.array([min(len(x[-3]), hist_len) for x in data["x_train"]])
            x_test["seq_length_s2"] = np.array([min(len(x[-3]), hist_len) for x in data["x_test"]])
            x_train["seq_length_hist_s1"] = np.array([min(len(x[-2]), hist_len) for x in data["x_train"]])
            x_test["seq_length_hist_s1"] = np.array([min(len(x[-2]), hist_len) for x in data["x_test"]])


        print("Saving..")
        data = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test, "feature_columns": feature_columns, "behavior_feature_list": behavior_feature_list}
        with open(processed_file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saving done.")

    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

    # Preprocess-input
    # Standarize
    if logarithm_input:
        for f in dense_feature_columns:
            name = f.name
            print(f"Putting log over {name}")
            x_train[name] = np.log(x_train[name] + 1)
            x_test[name] = np.log(x_test[name] + 1)
    if standarize_input:
        for f in dense_feature_columns:
            name = f.name
            print(f"Standarizing {name}")
            mean = np.mean(x_train[name], axis=0)
            std = np.std(x_train[name], axis=0)
            x_train[name] = (x_train[name] - mean) / std
            x_test[name] = (x_test[name] - mean) / std

    # Patch the embedding dim
    feature_columns = [f if isinstance(f, DenseFeat) else
            (f._replace(embedding_dim=emb_dim) if isinstance(f, SparseFeat) else
            f._replace(sparsefeat=f.sparsefeat._replace(embedding_dim=emb_dim))) for f in feature_columns]
    print(feature_columns)

    # Exclude some features
    if len(exclude) > 0:
        feature_columns = list(filter(lambda x: x.name != exclude, feature_columns))
    print(feature_columns)
    
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    return x_train, y_train, x_test, y_test, behavior_feature_list, feature_columns


def getMovieLensData(
        raw_path,
        o_filename,
        clk_thres=4, # clk_thres and above ratings are considered click
):
    movie_features = raw_path + "/movies.csv"
    interactions = raw_path + "/ratings.csv"

    movie_genres = {} # Key: movie_id, Val: list(movie cate)
    movie_id_remap = {}
    genre_id_remap = {}
    if path.exists(movie_features):
        print(f"Reading movie features from {movie_features}")
        with open(movie_features) as f:
            for j, line in enumerate(f):
                if j == 0:
                    continue
                parsed = line.split(",")
                movie_id = int(parsed[0])
                genres = parsed[-1].split("|")
                genres_new = []

                if movie_id not in movie_id_remap:
                    movie_id_remap[movie_id] = len(movie_id_remap) + 1 # 0 is reserved

                for genre in genres:
                    if genre not in genre_id_remap:
                        genre_id_remap[genre] = len(genre_id_remap) + 1 # 0 is reserved for unknown
                    genres_new.append(genre_id_remap[genre])
                assert(movie_id_remap[movie_id] not in movie_genres)
                movie_genres[movie_id_remap[movie_id]] = genres_new + []

        print(f"Read {j} movies. Without duplicate, {len(movie_genres)}")
    else:
        sys.exit(f"ERROR: {movie_features} not found")

    user_logs = {} # Key: uid, val: List[(timestamp, features)]

    counts = [0, 0, 0]
    data_train = {"X_cat": [], "y": []}
    data_test = {"X_cat": [], "y": []}

    if path.exists(interactions):
        print(f"Reading interactions from {interactions}")
        # Read through the samples to fill user logs
        with open(interactions) as f:
            for j, line in enumerate(f):
                if j == 0:
                    continue
                parsed = line.strip().split(",")
                user_id = int(parsed[0])
                movie_id = movie_id_remap[int(parsed[1])]
                rating = float(parsed[2])
                time = int(parsed[3])

                clk = 1 if rating >= clk_thres else 0

                assert(movie_id in movie_genres)

                if user_id not in user_logs:
                    user_logs[user_id] = []
                user_logs[user_id].append((time, clk, movie_id, movie_genres[movie_id] + []))

                # Update counts
                counts[0] = max(counts[0], user_id)
                counts[1] = max(counts[1], movie_id)
                counts[2] = max(counts[2], max(movie_genres[movie_id]))
            
            for j, uid in enumerate(user_logs):
                # For each entry, enrich the sparse feature with user history of
                # movie_id, move_cate_id.
                # For move_cate_id, we provide the list ordered with ascending popularity (not recentness).
                # To provide the full history, we need 2-D list which needs non-negligible code change.
                movie_hist = []
                genre_popularity = {}

                for i, (time, clk, movie_id, genre) in enumerate(sorted(user_logs[uid])):
                    # Train vs. Test split: 9:1
                    if i < len(user_logs[uid]) * 0.9:
                        data_train["y"].append(clk)
                        # Sort by popularity in ascending order
                        data_train["X_cat"].append((uid, movie_id, genre + [], movie_hist + [], [item[0] for item in sorted(genre_popularity.items(), key=lambda item: item[1])] + []))
                    else:
                        data_test["y"].append(clk)
                        data_test["X_cat"].append((uid, movie_id, genre + [], movie_hist + [], [item[0] for item in sorted(genre_popularity.items(), key=lambda item: item[1])] + []))

                    # Add history
                    if clk == 1:
                        movie_hist.append(movie_id)
                        for g in genre:
                            if g not in genre_popularity:
                                genre_popularity[g] = 0
                            genre_popularity[g] += 1
            print(f"Read {j} user logs")
    else:
        sys.exit(f"ERROR: {interactions} not found")

    # Add counts for the history features
    # TODO: This is currently hardcoded. May need to change if the feature structure changes
    counts += [counts[1], counts[2]]
    counts = [x + 1 for x in counts]
    print(len(data_train["y"]))
    print(len(data_test["y"]))

    data = {}
    data["y_train"] = data_train["y"]
    data["y_test"] = data_test["y"]
    data["x_train"] = data_train["X_cat"]
    data["x_test"] = data_test["X_cat"]

    # This is not exactly correct because the features that only the dropped (unknown) users
    # used can be removed without wasting embedding. However, we simply keep it.
    print(counts)
    data["counts"] = counts

    with open(o_filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Saved")

    return data


def getTaobaoAdsData(
        raw_path,
        o_filename,
        use_neg=False,
        include_timestamp=False
):
    raw_samples = raw_path + "/raw_sample.csv"
    ad_features = raw_path + "/ad_feature.csv"
    user_features = raw_path + "/user_profile.csv"

    ads = {} # Key: ad_id, val (List[dense], List[sparse])
    if path.exists(ad_features):
        print(f"Reading ad features from {ad_features}")
        with open(ad_features) as f:
            # Remove missing features and reorder
            # Instead of arbitrary renaming, trying to reorder only when necessary, with some extra work.
            tmp = []
            for j, line in enumerate(f):
                if j == 0:
                    continue
                parsed = line.split(",")
                if len(tmp) == 0:
                    for _ in range(len(parsed[:-1])):
                        tmp.append(set())
                for i, x in enumerate(parsed[:-1]):
                    tmp[i].add(int(x) if x != "NULL" else 0)
            rename_map = [{x:i for i, x in enumerate(sorted(s))} for s in tmp]

        with open(ad_features) as f:
            for j, line in enumerate(f):
                if j == 0:
                    continue
                parsed = line.split(",")
                assert(parsed[0] not in ads)
                # Only the last feature is dense (price)
                ads[int(parsed[0])] = ([float(parsed[-1])], [rename_map[i][int(x) if x != "NULL" else 0] for i, x in enumerate(parsed[:-1])])

        print(f"Read {j} ads. Without duplicate, {len(ads)}")
    else:
        sys.exit(f"ERROR: {ad_features} not found")

    users = {} # Key:user_id, val [List[dense], List[sparse]]
    if path.exists(user_features):
        print(f"Reading user features from {user_features}")
        with open(user_features) as f:
            # Remove missing features and reorder
            # Instead of arbitrary renaming, trying to reorder only when necessary, with some extra work.
            tmp = []
            for j, line in enumerate(f):
                if j == 0:
                    continue
                parsed = line.strip().split(",")
                if len(tmp) == 0:
                    for _ in range(len(parsed)):
                        tmp.append(set())
                for i, x in enumerate(parsed):
                    tmp[i].add(int(x) if x != "" else 0)
            rename_map = [{x:i for i, x in enumerate(sorted(s))} for s in tmp]

        with open(user_features) as f:
            for j, line in enumerate(f):
                if j == 0:
                    continue
                parsed = line.strip().split(",")
                assert(parsed[0] not in users)
                # No feature is dense
                users[int(parsed[0])] = ([], [rename_map[i][int(x) if x != "" else 0] for i, x in enumerate(parsed)])

        print(f"Read {j} users. Without duplicate {len(users)}")
    else:
        sys.exit(f"ERROR: {user_features} not found")

    data_by_day = {}

    cnt = 0
    # PID string to int
    pids = {}
    pid_cnt = 0

    user_logs = {} # Key: uid, val: List[(timestamp, features)]

    counts = [] # Count of distinct embeddings per feature
    if path.exists(raw_samples):
        print(f"Reading raw samples from {raw_samples}")
        # Read through the samples to fill user logs
        with open(raw_samples) as f:
            for j, line in enumerate(f):
                if j == 0:
                    continue
                parsed = line.strip().split(",")
                user_id = int(parsed[0])
                time = int(parsed[1])
                ad_id = int(parsed[2])
                if parsed[3] not in pids:
                    pids[parsed[3]] = pid_cnt
                    pid_cnt += 1
                pid = pids[parsed[3]]
                clk = int(parsed[5])
                # China time
                date = datetime.datetime.fromtimestamp(time, tz=datetime.timezone(datetime.timedelta(hours=8)))
                # Maybe using dayOfWeek is a bad idea because we only have 8 days of training.
                dayOfWeek = date.weekday()
                if (date.month, date.day) not in data_by_day:
                    data_by_day[(date.month, date.day)] = {"X_int": [], "X_cat": [], "y": []}

                if user_id in users and ad_id in ads:
                    if user_id not in user_logs:
                        user_logs[user_id] = []
                    # Save (time, y, X_int, X_cat)
                    if include_timestamp:
                        user_logs[user_id].append((time, clk, users[user_id][0] + ads[ad_id][0] + [time], users[user_id][1] + ads[ad_id][1] + [pid, dayOfWeek]))
                    else:
                        user_logs[user_id].append((time, clk, users[user_id][0] + ads[ad_id][0], users[user_id][1] + ads[ad_id][1] + [pid, dayOfWeek]))
                    cnt += 1
                    # Update counts
                    if len(counts) == 0:
                        counts = [0] * len(user_logs[user_id][-1][-1])
                    counts = [max(v1, v2) for v1, v2 in zip(counts, user_logs[user_id][-1][-1])]
            
            for j, uid in enumerate(user_logs):
                # For each entry, enrich the sparse feature with user history
                # Three user logs that are used by the DIN paper are:
                # ad id, category id, and brand id,
                # i.e., X_cat[ad_id_idx], X_cat[cate_id_idx], X_cat[brand_id_idx]
                # Note that their index calculation must change when the sparse feature structure changes
                ad_id_idx = len(users[uid][1])
                cate_id_idx = ad_id_idx + 1
                brand_id_idx = ad_id_idx + 4
                ad_hist = []
                cate_hist = []
                brand_hist = []

                if use_neg:
                    ad_hist_neg = []
                    cate_hist_neg = []
                    brand_hist_neg = []

                for time, y, X_int, X_cat in sorted(user_logs[uid]):
                    # China time
                    date = datetime.datetime.fromtimestamp(time, tz=datetime.timezone(datetime.timedelta(hours=8)))

                    # Aggregate features with user history and save it by day in case you want to do per-day shuffling
                    # We are ruining the original order, but it is a weird order anyways and we will shuffle it.
                    data_by_day[(date.month, date.day)]["y"].append(y)
                    data_by_day[(date.month, date.day)]["X_int"].append(X_int)
                    if use_neg:
                        data_by_day[(date.month, date.day)]["X_cat"].append(X_cat + [ad_hist + [], cate_hist + [], brand_hist + [], ad_hist_neg + [], cate_hist_neg + [], brand_hist_neg + []])
                    else:
                        data_by_day[(date.month, date.day)]["X_cat"].append(X_cat + [ad_hist + [], cate_hist + [], brand_hist + []])
                    
                    # Add history
                    if y == 1:
                        ad_hist.append(X_cat[ad_id_idx])
                        cate_hist.append(X_cat[cate_id_idx])
                        brand_hist.append(X_cat[brand_id_idx])

                        if use_neg:
                            ad_neg = random.randrange(counts[ad_id_idx] + 1)
                            while ad_neg in ad_hist:
                                ad_neg = random.randrange(counts[ad_id_idx] + 1)
                            ad_hist_neg.append(ad_neg)

                            cate_neg = random.randrange(counts[cate_id_idx] + 1)
                            while cate_neg in cate_hist:
                                cate_neg = random.randrange(counts[cate_id_idx] + 1)
                            cate_hist_neg.append(cate_neg)

                            brand_neg = random.randrange(counts[brand_id_idx] + 1)
                            while brand_neg in brand_hist:
                                brand_neg = random.randrange(counts[brand_id_idx] + 1)
                            brand_hist_neg.append(brand_neg)

    else:
        sys.exit(f"ERROR: {raw_samples} not found")

    # Add counts for the history features
    # TODO: This is currently hardcoded. May need to change if the feature structure changes
    counts += [counts[ad_id_idx], counts[cate_id_idx], counts[brand_id_idx]]
    if use_neg:
        counts += [counts[ad_id_idx], counts[cate_id_idx], counts[brand_id_idx]]
    counts = [x + 1 for x in counts]

    print({k: len(v["y"]) for k, v in data_by_day.items()})
    data = {"X_int": [], "X_cat": [], "y": []}
    total_per_file = []
    for date in sorted(data_by_day):
        d = data_by_day[date]
        data["y"] += d["y"]
        data["X_int"] += d["X_int"]
        data["X_cat"] += d["X_cat"]
        total_per_file.append(len(d["y"]))
    print(total_per_file)

    data["total_per_file"] = total_per_file

    # This is not exactly correct because the features that only the dropped (unknown) users
    # used can be removed without wasting embedding. However, we simply keep it.
    print(counts)
    data["counts"] = counts

    if include_timestamp:
        o_filename = o_filename.split(".")[0] + "_ts.pkl"
    with open(o_filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Saved")

    return data


if __name__ == "__main__":
    #getTaobaoAdsData("/checkpoint/kwmaeng/taobao/", "/checkpoint/kwmaeng/taobao/processed_tmp_test.pkl", False, False)
    getMovieLensData("/checkpoint/kwmaeng/movielens-20/", "/checkpoint/kwmaeng/movielens-20/processed_w_cate_5star.pkl", clk_thres=5)
