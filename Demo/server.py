import torch
import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_weights,
)

from collections import OrderedDict
import numpy as np
from typing import List, Optional, Tuple, Dict
import sys
import time
import datetime
import argparse
from pathlib import Path
import pandas as pd

from utils.util import (
    system_startup,
    set_random_seed,
    set_deterministic,
    Logger
)
from utils.resnetc import resnet20, resnet56
from utils.load_data import mysplit
import os
os.environ["GRPC_POLL_STRATEGY"] = "epoll1"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='test under federated learning framework')
    # Dataset
    parser.add_argument('--logs', default='./logs/', type=str)
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--ckp_path', default='./weights/ckp_r100.pth', type=str)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--method', default='iid', type=str)
    parser.add_argument('--rnd', default=1, type=int)
    parser.add_argument('--cnt', default=1, type=int)
    # server
    parser.add_argument('--strategy', default='random', type=str)
    parser.add_argument('--minfit', default=10, type=int)
    parser.add_argument('--mineval', default=10, type=int)
    parser.add_argument('--minavl', default=10, type=int)
    parser.add_argument('--num_rounds', default=100, type=int)
    # clients
    parser.add_argument('--subset', dest='subset', default='random', type=str)
    parser.add_argument('--TotalDevNum', dest='TotalDevNum', default=10, type=int)
    parser.add_argument('--unl_dev', default='2+8', type=str, help='devices which will ask to delete some data')
    # which metric to use
    parser.add_argument('--measure', default='tl', type=str, help='tl for translearn or ckp or vog or ps for poison')
    parser.add_argument('--unl_method', default='ori', type=str,
                        help='ori(no unlearn), base(train from scratch), cf (castrophic forgetting), ours (bi-level)')
    parser.add_argument('--unl_ratio', default=0.1, type=float)
    args = parser.parse_args()
    return args


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
        "local_epochs": 1,
        "lr": 0.1*(0.99**server_round), # for unlearn during the training
        # "lr": 0.01, # for extra unlearn after the training
    }
    return config


class SaveModelAndMetricsStrategy_random(fl.server.strategy.FedAvg):

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]], # FitRes is like EvaluateRes and has a metrics key
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        """Aggregate model weights using weighted average and store checkpoint"""
        aggregated_parameters_tuple = super().aggregate_fit(rnd, results, failures)
        aggregated_parameters, _ = aggregated_parameters_tuple

        if (rnd % Total_rnds == 0) and aggregated_parameters is not None:
            print(f"Saving round {rnd} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_weights: List[np.ndarray] = parameters_to_weights(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            torch.save(net.state_dict(), model_path + '_r' + str(rnd) + '_' + cnt_version + '.pth')

        return aggregated_parameters_tuple

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # for r in results:
        #     print(r[0].cid, r[1].metrics['unl_acc'])

        # get accuracy of each client
        accuracies = list(map(lambda r: r[1].metrics['accuracy'], results))
        examples = list(map(lambda r: r[1].num_examples, results))
        losses = list(map(lambda r: r[1].loss, results))
        # record the fl_acc and unl_acc
        flacc_list.append(accuracies[0])
        cid_list = list(map(lambda r: r[1].metrics['cid'], results))
        if use_measure == 'none':
            accuracies_unl = list(map(lambda r: 0., results))
            efficacy_unl = list(map(lambda r: 0., results))
            examples_unl = list(map(lambda r: 0., results))
        else:
            accuracies_unl = list(map(lambda r: r[1].metrics['unl_acc'], results))
            efficacy_unl = list(map(lambda r: r[1].metrics['unl_efficacy'], results))
            examples_unl = list(map(lambda r: r[1].metrics['unlearn_imgs'], results))
        print('Round {:03d} | FL_loss {:.4f} | FL_acc {:.4f} | Test images {} | Unl_acc {:.4f} | Efficacy_acc {:.4f} | Unl images {}'.format(
            rnd, losses[0], accuracies[0], examples[0], accuracies_unl[0], efficacy_unl[0], examples_unl[0]))
        for i in range(Total_devices):
            # # update with all clients
            # unlacc_list[cid_list[i]-1].append(accuracies_unl[i])
            # effacc_list[cid_list[i]-1].append(efficacy_unl[i])
            # update with only unlearned clients
            if cid_list[i] == 2:
                unlacc_list[0].append(accuracies_unl[i])
                effacc_list[0].append(efficacy_unl[i])
            elif cid_list[i] == 8:
                unlacc_list[1].append(accuracies_unl[i])
                effacc_list[1].append(efficacy_unl[i])
            # elif cid_list[i] == 6:
            #     unlacc_list[2].append(accuracies_unl[i])
            #     effacc_list[2].append(efficacy_unl[i])
            # elif cid_list[i] == 8:
            #     unlacc_list[3].append(accuracies_unl[i])
            #     effacc_list[3].append(efficacy_unl[i])

        if rnd == st_rnd:
            self.best_acc = 0.
            self.best_rnd = 0
        if self.best_acc < accuracies[0]:
            self.best_acc = accuracies[0]
            self.best_rnd = rnd
        if rnd == Total_rnds:
            print(f'Best FL accuracy {self.best_acc} on round {self.best_rnd}')

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)


if __name__ == '__main__':

    args = parse_args()
    # set save path
    unl_dev_list = mysplit(args.unl_dev)
    dev_name = 'c'
    cnt_version = str(args.cnt)
    for i in range(len(unl_dev_list)):
        dev_name = dev_name + str(unl_dev_list[i]) + '_c'
    save_path = args.logs + args.dataset + '/' + args.measure + '/' + dev_name + '/' + str(args.unl_ratio)
    model_path = save_path + '/weights/ckp_' + args.unl_method
    Path(save_path + '/weights').mkdir(parents=True, exist_ok=True)
    Path(save_path + '/server_logs').mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(save_path + '/server_logs/' + args.unl_method + '_sever_' + cnt_version + '.csv', sys.stdout)

    print(args)
    # Choose GPU device and print status information
    device, setup = system_startup()
    set_random_seed(args.seed)
    set_deterministic()

    if args.measure == 'tl':
        use_measure = 'translearn'
        print('Use samples with new cls to measure the unlearning effect.')
    elif args.measure == 'ckp':
        use_measure = 'ckp'
        print('Use samples with high misclassification rates for the ckp to measure the unlearning effect.')
    elif args.measure == 'vog':
        use_measure = 'vog'
        print('Use samples with high vog to measure the unlearning effect.')
    elif args.measure == 'ps':
        use_measure = 'poison'
        print('Use poison samples to measure the unlearning effect.')
    elif args.measure == 'none':
        use_measure = 'none'
        print('Use normal training.')
    else:
        assert False, 'not support other measurement yet.'

    # Load Model
    if args.dataset == 'CIFAR10':
        net = resnet20()
    elif args.dataset == 'CIFAR100':
        net = resnet56(num_classes=100)
    net.to(**setup)

    Total_devices = args.mineval
    Total_rnds = args.num_rounds
    st_rnd = args.rnd
    flacc_list = []
    unlacc_list = [[] for i in range(Total_devices)]
    effacc_list = [[] for i in range(Total_devices)]
    mydict = {}
    if args.pretrained:
        # Load Pre-trained Model
        ckp = torch.load(args.ckp_path)
        init_params = [val.cpu().numpy() for _, val in ckp.items()]

        # Create strategy and run server
        strategy = SaveModelAndMetricsStrategy_random(
            fraction_fit=0.1, # Sample 10% of available clients for the next round
            min_fit_clients=args.minfit, # Minimum number of clients to be sampled for the next round
            fraction_eval=0.01, # Fraction of clients used during validation
            min_eval_clients=args.mineval, # Minimum number of clients used during validation
            min_available_clients=args.minavl, # Minimum number of clients that need to be connected to the server before a training round can start
            on_fit_config_fn=fit_config, # Function that returns the training configuration for each round
            initial_parameters=init_params, # Initial model parameters
        )
    else:
        # Create strategy and run server
        strategy = SaveModelAndMetricsStrategy_random(
            fraction_fit=0.1, # Sample 10% of available clients for the next round
            min_fit_clients=args.minfit, # Minimum number of clients to be sampled for the next round
            fraction_eval=0.01, # Fraction of clients used during validation
            min_eval_clients=args.mineval, # Minimum number of clients used during validation
            min_available_clients=args.minavl, # Minimum number of clients that need to be connected to the server before a training round can start
            on_fit_config_fn=fit_config, # Function that returns the training configuration for each round
        )
    start_time = time.time()
    fl.server.start_server("[::]:8080", config={"num_rounds": args.num_rounds}, strategy=strategy)

    # Save the results
    mydict.update({'S_acc': flacc_list})
    for i in range(Total_devices):
        if (not args.pretrained) or (len(unl_dev_list) < Total_devices):
            mydict.update({'C'+str(i+1)+'_acc': unlacc_list[i]})
            mydict.update({'C'+str(i+1)+'_efficacy': effacc_list[i]})
        else:
            mydict.update({'C'+str(unl_dev_list[i])+'_acc': unlacc_list[i]})
            mydict.update({'C'+str(unl_dev_list[i])+'_efficacy': effacc_list[i]})
    dataframe = pd.DataFrame(mydict)
    dataframe.to_csv(save_path + '/' + args.unl_method + '_' + cnt_version + '.csv', index=False, sep=',')

    # Print final timestamp
    print('Strategy Method: ', args.strategy)
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print(f"Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}")
    print()
