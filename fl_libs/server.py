import numpy as np
import copy

import torch
import torch.distributed as dist

from sklearn.metrics import log_loss, roc_auc_score, recall_score, precision_score, f1_score, average_precision_score
import time

import math
import random
import pickle

from .utils import set_optimizer

class Server:
    def __init__(self, rank, world_size, train_clients, test_data, model, device, optimizer, server_lr, total_train_cids, test_metrics, seed, is_heterofl=False, overselect=0.0):
        # Make clients a dict. That is more handy
        # TODO: This needs some cleanup
        self.local_train_clients = {c.cid: c for c in train_clients}
        self.local_total_clients = {c.cid: c for c in train_clients}

        self.total_train_cids = total_train_cids

        self.test_data = test_data

        self.model = copy.deepcopy(model)
        self.device = device
        self.total_weight = 0
        self.server_lr = server_lr
        self.optim = set_optimizer(self.model, optimizer, server_lr)
        self.rank = rank
        self.world_size = world_size

        self.seed = seed

        self.test_metrics = [x.strip() for x in test_metrics.split(",")]
        self.is_heterofl = is_heterofl

        self.overselect = overselect


    def init_clients(self, model, device, optimizer):
        for cid, c in self.local_total_clients.items():
            c.init_client(model, device, optimizer)


    def select_clients(self, clients_per_round, r, client_selection, overselect):
        client_size = min(int(clients_per_round * (1 + overselect)), len(self.total_train_cids))

        if client_selection == "random":
            np.random.seed(r)
            selected_clients = np.random.choice(self.total_train_cids, client_size, replace=False)
        elif client_selection == "sequential":
            selected_clients = self.total_train_cids[r * client_size : (r + 1) * client_size]

        # Remove slowest x% from the overselection. According to the data from FLASH,
        # low <<< mid <<< hi, so we can just remove from low tier.

        if overselect > 0:
            hi_clients = list(filter(lambda c: c[1] == 2, selected_clients))
            mid_clients = list(filter(lambda c: c[1] == 1, selected_clients))
            low_clients = list(filter(lambda c: c[1] == 0, selected_clients))
            clients_to_remove = min(len(selected_clients), int(clients_per_round * overselect))
            if len(low_clients) > clients_to_remove:
                low_clients = low_clients[:len(low_clients) - clients_to_remove]
            else:
                clients_to_remove -= len(low_clients)
                low_clients = []
                if len(mid_clients) > clients_to_remove:
                    mid_clients = mid_clients[:len(mid_clients) - clients_to_remove]
                else:
                    clients_to_remove -= len(mid_clients)
                    mid_clients = []
                    if len(hi_clients) > clients_to_remove:
                        hi_clients = hi_clients[:len(hi_clients) - clients_to_remove]
                    else:
                        hi_clients = []
            selected_clients = low_clients + mid_clients + hi_clients

        if len(selected_clients) == 0:
            return None # All clients used
        return [self.local_train_clients[cid] for cid, _ in list(filter(lambda x: x[0] in self.local_train_clients, selected_clients))]


    def train(self, clients_per_round, epochs, bs, aggr_method, test_freq, client_lr, client_selection="random"):
        timestamps = []
        print("Start training", flush=True)

        r = 0
        while True:
            timestamps.append(time.time())
            if ((r + 1) % test_freq == 0) and self.rank == 0:
                print(f"========== Round {r} ===========", flush=True)
            # Model to train mode if not
            self.model = self.model.train()

            # Select clients
            selected_clients = self.select_clients(clients_per_round, r, client_selection, self.overselect)

            if selected_clients is None:
                # All clients used
                print(f"All clients used", flush=True)
                return

            #print(f"Round {r}: selected clients for rank {self.rank}: {[c.cid for c in selected_clients]}")

            # Temporary dict to accumulate gradients
            accum = {name: torch.zeros(param.shape if len(param.shape) > 0 else 1, dtype=param.dtype).to(self.device) for name, param in self.model.state_dict().items()}

            # For sparse embedding features, we count the # samples per row (for FSL).
            # We can optimize this further, as all dense features share the same count. However, we do not do such an optimization for now.
            total_samples = {name: torch.zeros(param.shape[0]).to(self.device) if ("embedding_dict" in name and aggr_method == "fsl") else (torch.zeros_like(param) if (("dnn.linears" in name or "dnn_linear" in name) and self.is_heterofl) else torch.zeros(1)).to(self.device) for name, param in self.model.state_dict().items()}

            for c in selected_clients:
                # Downstream the model
                c.model.load_state_dict(self.model.state_dict())

                # Run local training
                num_samples, update = c.train(epochs, bs, client_lr)

                for name in accum:
                    # Collect the gradients
                    accum[name] += update[name] * num_samples

                    # Count total # of samples, handling sparse features with care
                    if "embedding_dict" in name and aggr_method == "fsl":
                        total_samples[name] += (update[name][:, 0] != 0.0) * num_samples
                    # For HeteroFL aggregation.
                    elif ("dnn.linears" in name or "dnn_linear" in name) and self.is_heterofl:
                        # Currently, only hardcoded for 2D tensors
                        assert(len(update[name].shape) == 2)
                        row_bnd = math.ceil(update[name].shape[0] * c.channel_multiplier)
                        col_bnd = math.ceil(update[name].shape[1] * c.channel_multiplier)
                        mask = torch.zeros_like(update[name]).to(self.device)
                        if "0.weight" in name:
                            assert(torch.sum(update[name][row_bnd:, :]) == 0.0)
                            # First DNN layer
                            mask[:row_bnd,:] = 1.0
                        else:
                            assert(torch.sum(update[name][row_bnd:, :]) == 0.0 and torch.sum(update[name][:, col_bnd:]) == 0.0)
                            # All other layers and bias
                            mask[:row_bnd,:col_bnd] = 1.0
                        total_samples[name] += mask * num_samples
                    else:
                        total_samples[name] += num_samples

            # Total_samples zero to 1 to avoid divide by zero
            if aggr_method == "fsl":
                for name in total_samples:
                    if "embedding_dict" in name:
                        total_samples[name][total_samples[name] == 0.0] = 1.0

            # All-reduce the accumulated gradients and the sample stats across GPUs
            reqs = []
            for name in accum:
                reqs.append(dist.all_reduce(total_samples[name], async_op=True))
                reqs.append(dist.all_reduce(accum[name], async_op=True))
            for req in reqs:
                req.wait()

            # Normalize the gradients
            for ii, name in enumerate(accum):
                if "embedding_dict" in name and aggr_method == "fsl":
                    accum[name] /= total_samples[name][:, None]
                # Handling BN (I handle it naively)
                elif "num_batches_tracked" in name:
                    accum[name] //= total_samples[name].long()
                else:
                    accum[name] /= total_samples[name]

            # Feed the aggregated gradient to the server optimizer
            updated = set()
            for name, param in self.model.named_parameters():
                param.grad = accum[name].detach().clone()
                updated.add(name)

            # Step
            self.optim.step()

            # Update other states not handled by the optimizer, such as batch statistics
            with torch.no_grad():
                for name, param in self.model.state_dict().items():
                    if name not in updated:
                        if "num_batches_tracked" in name:
                            param += (self.server_lr * accum[name][0]).type(param.dtype)
                        else:
                            param += self.server_lr * accum[name]

            if ((r + 1) % test_freq == 0):
                if self.rank == 0:
                    print(f"Average time/round", np.mean([timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]))
                    timestamps = []
                    self.test()

            r += 1


    def test(self, save_result=False):
        total_ans = np.array([])
        total_y = np.array([])
        t = time.time()
        for tier in self.test_data:
            x_test = self.test_data[tier]["x_test"]
            y_test = self.test_data[tier]["y_test"]
            pred_ans = self.model.predict(x_test, 256)
            self.print_metrics(y_test, pred_ans, f"Tier {tier}")
            total_ans = np.concatenate([total_ans, pred_ans.squeeze()])
            total_y = np.concatenate([total_y, y_test.squeeze()])
            if save_result:
                print(f"Saving for tier {tier} {t}...")
                with open(f"./y_test_{tier}_{t}.pkl", 'wb') as handle:
                    pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(f"./pred_ans_{tier}_{t}.pkl", 'wb') as handle:
                    pickle.dump(pred_ans, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.print_metrics(total_y, total_ans, f"Total")
        print(f"Total test loss", round(log_loss(total_y, total_ans, labels=[0, 1]), 4))


    def print_metrics(self, y, ans, prefix=""):
        for metric in self.test_metrics:
            if metric == "loss":
                print(f"{prefix} test loss", round(log_loss(y, ans, labels=[0, 1]), 4), flush=True)
            elif metric == "auc":
                if 0 < np.sum(y) < len(y):
                    print(f"{prefix} test AUC", round(roc_auc_score(y, ans), 4), flush=True)
                else:
                    print("Cannot calculate AUC")
            elif metric == "recall":
                # TODO: Assuming 0.5 cutoff
                print(f"{prefix} test Recall", round(recall_score(y, np.round(ans, 0), labels=[0, 1]), 4), flush=True)
            elif metric == "precision":
                print(f"{prefix} test Precision", round(precision_score(y, np.round(ans, 0), labels=[0, 1]), 4), flush=True)
            elif metric == "f1":
                print(f"{prefix} test f1", round(f1_score(y, np.round(ans, 0), labels=[0, 1]), 4), flush=True)
            elif metric == "ap":
                print(f"{prefix} test ap", round(average_precision_score(y, ans), 4), flush=True)
            else:
                print(f"{prefix} Unknown metric {metric}")
