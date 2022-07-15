import argparse

def get_args():
    ### parse arguments ###
    parser = argparse.ArgumentParser()

    ## common model-related parameters
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--model", type=str, default="din")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hist-len", type=int, default=10)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--l2-reg-dnn", type=float, default=0.0)
    parser.add_argument("--l2-reg-emb", type=float, default=1e-6)
    parser.add_argument("--l2-reg-linear", type=float, default=0.0)
    parser.add_argument("--combiner", type=str, default="mean")

    # Common system related params
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--metrics", type=str, default="loss,auc,ap")

    # For DNN-base models (for DLRM, this is the top MLP)
    parser.add_argument("--dnn-hidden-units", type=str, default="256")

    # For DIN/DIEN
    parser.add_argument("--att-hidden-units", type=str, default="64-16")
    parser.add_argument("--dice-norm-type", type=str, default=None)
    parser.add_argument("--att-activation", type=str, default="Dice")
    parser.add_argument("--att-weight-normalization", action="store_true", default=False)

    # For DCNv1/v2
    parser.add_argument("--cross-num", type=int, default=2)

    # For DLRM
    parser.add_argument("--bot-dnn-hidden-units", type=str, default="16")

    ## FL-related params
    parser.add_argument("--fl", action="store_true", default=False)
    parser.add_argument("--aggr-method", type=str, default="fedavg")
    parser.add_argument("--min-sample-size", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=10)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--server-optimizer", type=str, default="adagrad")
    parser.add_argument("--client-optimizer", type=str, default="sgd")
    parser.add_argument("--data-split", type=str, default="niid")
    parser.add_argument("--num-clients", type=int, default=0)
    parser.add_argument("--server-lr", type=float, default=0.01)
    parser.add_argument("--num-global-epochs", type=int, default=1)

    # Input preprocessing
    parser.add_argument("--dataset", type=str, default="taobao")
    parser.add_argument("--logarithm-input", action="store_true", default=False)
    parser.add_argument("--standarize-input", action="store_true", default=False)
    parser.add_argument("--target-ctr", type=float, default=-1)

    # Exclude a certain feature. For DIN/DLRM, we exclude s0 (user id)
    parser.add_argument("--exclude", type=str, default="s0")

    # Tier-aware optimizations
    parser.add_argument("--tier-selection", type=str, default="random")
    parser.add_argument("--exclude-low", action="store_true", default=False)
    parser.add_argument("--overselect", type=float, default=0.0)
    parser.add_argument("--comm-prune-rate", type=str, default="0.0") # Can be in the form of "0.7-0.5-0.0" to use different params for lo-mid-high end devices
    parser.add_argument("--comm-quantize-bits", type=str, default="0") # Can be used in the form of "2-8-0", 0 is no quantization.
    parser.add_argument("--comm-quantize-separate-sign", action="store_true", default=False)
    parser.add_argument("--heterofl-channel-multiplier", type=str, default="1.0") # Channel multiplier per tiers. e.g., "0.25-0.5-1.0"

    return parser.parse_args()
