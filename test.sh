# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# FL: MovieLens-20
python3.8 run.py --dataset movielens-20 --model dlrm --logarithm-input --att-weight-normalization --fl --learning-rate 1.0 --client-optimizer sgd --batch-size 0 --server-lr 1e-1 --test-freq 100 --ngpus 2 --clients-per-round 100 --tier-selection dirichlet-s1-0.005 --seed 123 --exclude-low

# FL: Taobao
#python3.8 run.py --dataset taobao --model dlrm --logarithm-input --att-weight-normalization --fl --learning-rate 1.0 --client-optimizer sgd --batch-size 0 --server-lr 1e-2 --test-freq 1000 --ngpus 2 --clients-per-round 100 --tier-selection dirichlet-s9-0.005 --seed 123

# Non-FL: MovieLens-20
#python3.8 run.py --dataset movielens-20 --model dlrm --logarithm-input --att-weight-normalization --learning-rate 1e-1 --client-optimizer adagrad --batch-size 128 --test-freq 100 --ngpus 2 --seed 123

# Non-FL: Taobao
#python3.8 run.py --dataset taobao --model dlrm --logarithm-input --att-weight-normalization --learning-rate 1e-2 --client-optimizer adagrad --batch-size 128 --test-freq 1000 --ngpus 2 --seed 123
