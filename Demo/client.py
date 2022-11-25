import torch
import argparse
import sys
import time
import datetime
import flwr as fl
from pathlib import Path
from collections import OrderedDict
from PIL import Image
import numpy as np

from utils.util import (
    system_startup,
    set_random_seed,
    set_deterministic,
    Logger
)
from utils.load_data import (
    mysplit,
    load_data_tl,
    load_data_vog,
    load_data_poison,
    load_data_ori,
    load_data_ckp
)
from utils.learn import train, test, train_unl, add_unl_param, get_ori_param
from utils.sam import SAM
from utils.resnetc import resnet20, resnet56
from utils.ada_hessain import AdaHessian

import os
os.environ["GRPC_POLL_STRATEGY"] = "epoll1"


def parse_args():

    parser = argparse.ArgumentParser(description='Test using clients')
    # Dataset
    parser.add_argument('--logs', default='./logs/', type=str)
    parser.add_argument('--root', default='/mnt/data/dataset', type=str)
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--method', default='iid', type=str)
    parser.add_argument('--train_lr', default=0.1, type=float)
    parser.add_argument('--rnd', default=1, type=int)
    # clients
    parser.add_argument('--subset', dest='subset', default='random', type=str)
    parser.add_argument('--TotalDevNum', dest='TotalDevNum', default=10, type=int)
    parser.add_argument('--DevNum', dest='DevNum', default=1, type=int)
    # unlearn
    parser.add_argument('--unl_ratio', default=0.1, type=float)
    parser.add_argument('--unl_method', default='ori', type=str,
                        help='ori(no unlearn), base(train from scratch), cf (castrophic forgetting), ours (bi-level), frr (federated rapid retraining)')
    parser.add_argument('--unl_dev', default='2+8', type=str, help='devices which will ask to delete some data')
    parser.add_argument('--unl_rnd', default=80, type=int, help='the round when the device ask to delete')
    parser.add_argument('--unl_endrnd', default=89, type=int, help='the round when our method finish')
    parser.add_argument('--version', default='minmax', type=str, help='maxmin, minmax')
    parser.add_argument('--K_steps', default=4, type=int, help='K steps for LL')
    parser.add_argument('--S_steps', default=8, type=int, help='S steps for UL')
    parser.add_argument('--alpha', default=0.001, type=float, help='learning rate for UL')
    parser.add_argument('--loss_thr', default=10., type=float)
    parser.add_argument('--use_comp', default=False, action='store_true', help='compensate for unlearning')
    parser.add_argument('--gamma', default=1, type=float, help='weight for adding unl_param')
    # SAM for unlearning
    parser.add_argument('--use_sam', default=False, action='store_true')
    parser.add_argument("--rho", default=0.05, type=int,
                        help="Rho parameter for SAM.")
    parser.add_argument("--adaptive", default=False, type=bool,
                        help="True if you want to use the Adaptive SAM.")
    # which metric to use
    parser.add_argument('--measure', default='tl', type=str, help='tl for translearn or ckp or vog or ps for poison')
    parser.add_argument('--ckp_num', default=60, type=int)
    # transfer learning (samples from new cls) for verifying unlearn effect
    parser.add_argument('--new_label', default='1+9', type=str)
    # use normalized vog or not normalized vog
    parser.add_argument('--normalized_vog', default=False, action='store_true')
    # poison samples for verify unlearn effect
    parser.add_argument('--tar_label', default=9, type=int)
    parser.add_argument('--j0', type=float, default=10, help="")
    parser.add_argument('--T0', type=float, default=5, help="")
    parser.add_argument('--frr_lr', type=float, default=0.1, help="")
    args = parser.parse_args()
    return args


class MyClient_tl(fl.client.NumPyClient):
    def __init__(self, net, args, device, setup):
        super(MyClient_tl, self).__init__()
        self.net = net
        self.args = args
        self.device = device
        # load data
        self.trainloader_all, self.trainloader_ideal, self.unlearned_trainloader, \
            self.testloader, self.num_examples, self.testloader_newcls = load_data_tl(args)
        self.dm = torch.as_tensor(self.num_examples['dm'], **setup)[None, :, None, None]  # 1xCx1x1
        self.ds = torch.as_tensor(self.num_examples['ds'], **setup)[None, :, None, None]
        # define loss function and optimizer
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.train_lr, momentum=0.9, weight_decay=5e-4)  # for normal training
        self.optimizer_un = torch.optim.SGD(self.net.parameters(), lr=args.alpha, momentum=0.9, maximize=True)  # for unlearning UL
        optimizer_sam = SAM(self.net.parameters(), torch.optim.SGD, rho=args.rho, adaptive=args.adaptive,
                            lr=args.alpha, momentum=0.9, maximize=True)
        if args.use_sam:
            self.optimizer_un = optimizer_sam
        else:
            self.optimizer_un = self.optimizer_un
        if self.args.unl_method == 'frr':
            self.optimizer_frr = AdaHessian(self.net.parameters(), lr=args.frr_lr)
        self.unl_dev_list = mysplit(args.unl_dev)

    def get_parameters(self):
        '''return the model weight as a list of NumPy ndarrays'''
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        '''update the local model weights with the parameters received from the server'''
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        '''set the local model weights, train the local model,
           receive the updated local model weights'''
        self.set_parameters(parameters)
        self.optimizer.param_groups[0]['lr'] = config['lr']

        # train the local model
        if (self.args.DevNum in self.unl_dev_list) and self.args.unl_method != 'base' and self.args.unl_method != 'frr':
            train_num = self.num_examples["trainset"]

            # # save images for gradcam demo
            # for batch_idx, (images, targets) in enumerate(self.trainloader_ideal):
            #     images, targets = images.to(self.device), targets.to(self.device)
            #     if batch_idx == 0:
            #         for indx in range(images.size(0)):
            #             remain_img = Image.fromarray(((images*self.ds+self.dm)[indx, :].clip(0, 1).cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8))
            #             remain_img.save('./logs/CIFAR10/images/tl/ur0.01/c{}/remain_{}_y{}.png'.format(self.args.DevNum, indx, targets[indx]))
            #         break
            # for batch_idx, (images, targets) in enumerate(self.unlearned_trainloader):
            #     images, targets = images.to(self.device), targets.to(self.device)
            #     if batch_idx == 0:
            #         for indx in range(images.size(0)):
            #             unl_img = Image.fromarray(((images*self.ds+self.dm)[indx, :].clip(0, 1).cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8))
            #             unl_img.save('./logs/CIFAR10/images/tl/ur0.01/c{}/unl_{}_y{}.png'.format(self.args.DevNum, indx, targets[indx]))
            #         break

            # the round before the unlearning or have no unlearning process
            if (config['current_round'] < self.args.unl_rnd) or (self.args.unl_method == 'ori'):
                train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer, self.net,
                                self.trainloader_all, self.device)
                print('Device-{:02d} | Round-{:03d} | Train Loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                    self.args.DevNum, config['current_round'], train_loss,
                    self.optimizer.param_groups[0]['lr'], self.num_examples["trainset"]))

            elif self.args.unl_method == 'cf':
                train_num = self.num_examples["ideal_trainset"]
                # the round after the client ask to delete some data, use the normal training on the remaining data
                train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer,
                                   self.net, self.trainloader_ideal, self.device)
                print('CF | Unlearned Device-{:02d} | Round-{:03d} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                    self.args.K_steps, self.args.DevNum, config['current_round'], train_loss,
                    self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))

            elif self.args.unl_method == 'ours':
                train_num = self.num_examples["ideal_trainset"]
                # the round when the client ask to delete some data and use our methods for t rounds to unlearn
                if (config['current_round'] >= self.args.unl_rnd) and (config['current_round'] <= self.args.unl_endrnd):

                    ori_param = get_ori_param(self.net)
                    for i in range(self.args.S_steps):
                        # get \theta_LL^k for LL: unlearn the data
                        for j in range(self.args.K_steps):
                            unlearn_loss, unl_param = train_unl(self.loss_fn, self.optimizer_un, self.net,
                                    self.unlearned_trainloader, self.device, self.args.loss_thr, self.args.measure,
                                    self.args.tar_label, self.args.use_sam)
                        # get \theta_UL^i for UL: train on the remaining data
                        train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer,
                                        self.net, self.trainloader_ideal, self.device)

                    print('Ours S{}K{} ({}) | Unlearn Device-{:02d} | Round-{:03d} | Unlearn_loss-{:.4f} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                        self.args.S_steps, self.args.K_steps, self.args.version, self.args.DevNum, config['current_round'], unlearn_loss, train_loss,
                        self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))
                    # compensate the unlearning param
                    if self.args.use_comp:
                        add_unl_param(self.net, ori_param, unl_param, self.args.gamma)
                # the round after our methods, use the normal training on the remaining data
                elif config['current_round'] > self.args.unl_endrnd:
                    unl_param = None
                    train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer,
                                    self.net, self.trainloader_ideal, self.device)
                    print('Ours S{}K{} | Unlearned Device-{:02d} | Round-{:03d} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                        self.args.S_steps, self.args.K_steps, self.args.DevNum, config['current_round'], train_loss,
                        self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))

        # train form scratch using rapid retraining
        elif self.args.unl_method == 'frr':
            train_num = self.num_examples["ideal_trainset"]
            if ((config['current_round'] - self.args.j0) % self.args.T0 ==0) and (config['current_round'] <= self.args.j0):
                optimizer_frr = self.optimizer
                retrain = False
            else:
                optimizer_frr = self.optimizer_frr
                retrain = True
            # optimizer_frr.param_groups[0]['lr'] = config['lr']
            train_loss = train(config["local_epochs"], self.loss_fn, optimizer_frr,
                         self.net, self.trainloader_ideal, self.device, retrain=retrain).to(self.device)
            print('FRR | Device-{:02d} | Round-{:03d} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                  self.args.DevNum, config['current_round'], train_loss,
                  self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))

        # the client is not in the unlearned device list or train form scratch
        else:
            train_num = self.num_examples["ideal_trainset"]
            train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer,
                         self.net, self.trainloader_ideal, self.device)
            print('Device-{:02d} | Round-{:03d} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                  self.args.DevNum, config['current_round'], train_loss,
                  self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))

        return self.get_parameters(), train_num, {"loss": float(train_loss)}

    def evaluate(self, parameters, config):
        '''test the local model'''
        self.set_parameters(parameters)
        loss, accuracy, _ = test(self.loss_fn, self.net, self.testloader, self.device)
        _, acc_tl, efficacy_tl = test(self.loss_fn, self.net, self.unlearned_trainloader, self.device, use_score=True)
        print('Device-{:02d} | Test Loss-{:.4f} | Accuracy-{:.2f} | Images-{:} | TL Accuracy-{:.4f} | TL Efficacy-{:.4f} | TL Images-{:}'.format(
            self.args.DevNum, loss, accuracy, self.num_examples["testset"], acc_tl, efficacy_tl, self.num_examples["unlearned_trainset"]))
        return float(loss), self.num_examples["testset"], \
            {"cid": self.args.DevNum, "accuracy": float(accuracy), "unl_acc": float(acc_tl), \
             "unl_efficacy": float(efficacy_tl), "unlearn_imgs": self.num_examples["unlearned_trainset"]}

class MyClient_ckp(fl.client.NumPyClient):
    def __init__(self, net, args, device, setup):
        super(MyClient_ckp, self).__init__()
        self.net = net
        self.args = args
        self.device = device
        # load data
        self.trainloader_all, self.trainloader_ideal, self.unlearned_trainloader, \
            self.testloader, self.num_examples = load_data_ckp(args)
        self.dm = torch.as_tensor(self.num_examples['dm'], **setup)[None, :, None, None]  # 1xCx1x1
        self.ds = torch.as_tensor(self.num_examples['ds'], **setup)[None, :, None, None]
        # define loss function and optimizer
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.train_lr, momentum=0.9, weight_decay=5e-4)  # for normal training
        self.optimizer_un = torch.optim.SGD(self.net.parameters(), lr=args.alpha, momentum=0.9, maximize=True)  # for unlearning UL
        optimizer_sam = SAM(self.net.parameters(), torch.optim.SGD, rho=args.rho, adaptive=args.adaptive,
                            lr=args.alpha, momentum=0.9, maximize=True)
        if args.use_sam:
            self.optimizer_un = optimizer_sam
        else:
            self.optimizer_un = self.optimizer_un
        if self.args.unl_method == 'frr':
            self.optimizer_frr = AdaHessian(self.net.parameters(), lr=args.frr_lr, weight_decay=1e-6)
        self.unl_dev_list = mysplit(args.unl_dev)

    def get_parameters(self):
        '''return the model weight as a list of NumPy ndarrays'''
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        '''update the local model weights with the parameters received from the server'''
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        '''set the local model weights, train the local model,
           receive the updated local model weights'''
        self.set_parameters(parameters)
        self.optimizer.param_groups[0]['lr'] = config['lr']

        # train the local model
        if (self.args.DevNum in self.unl_dev_list) and self.args.unl_method != 'base' and self.args.unl_method != 'frr':
            train_num = self.num_examples["trainset"]

            # # save images for gradcam demo
            # for batch_idx, (images, targets) in enumerate(self.trainloader_ideal):
            #     images, targets = images.to(self.device), targets.to(self.device)
            #     if batch_idx == 0:
            #         for indx in range(images.size(0)):
            #             remain_img = Image.fromarray(((images*self.ds+self.dm)[indx, :].clip(0, 1).cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8))
            #             remain_img.save('./logs/CIFAR10/images/ckp/ur0.01/c{}/remain_{}_y{}.png'.format(self.args.DevNum, indx, targets[indx]))
            #         break
            # for batch_idx, (images, targets) in enumerate(self.unlearned_trainloader):
            #     images, targets = images.to(self.device), targets.to(self.device)
            #     if batch_idx == 0:
            #         for indx in range(images.size(0)):
            #             unl_img = Image.fromarray(((images*self.ds+self.dm)[indx, :].clip(0, 1).cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8))
            #             unl_img.save('./logs/CIFAR10/images/ckp/ur0.01/c{}/unl_{}_y{}.png'.format(self.args.DevNum, indx, targets[indx]))
            #         break

            # the round before the unlearning or have no unlearning process
            if (config['current_round'] < self.args.unl_rnd) or (self.args.unl_method == 'ori'):
                train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer, self.net,
                                   self.trainloader_all, self.device)
                print('Device-{:02d} | Round-{:03d} | Train Loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                    self.args.DevNum, config['current_round'], train_loss,
                    self.optimizer.param_groups[0]['lr'], self.num_examples["trainset"]))

            elif self.args.unl_method == 'cf':
                train_num = self.num_examples["ideal_trainset"]
                # the round after the client ask to delete some data, use the normal training on the remaining data
                train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer,
                                   self.net, self.trainloader_ideal, self.device)
                print('CF | Unlearned Device-{:02d} | Round-{:03d} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                    self.args.K_steps, self.args.DevNum, config['current_round'], train_loss,
                    self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))

            elif self.args.unl_method == 'ours':
                train_num = self.num_examples["ideal_trainset"]
                # the round when the client ask to delete some data and use our methods for t rounds to unlearn
                if (config['current_round'] >= self.args.unl_rnd) and (config['current_round'] <= self.args.unl_endrnd):

                    ori_param = get_ori_param(self.net)
                    for i in range(self.args.S_steps):
                        # get \theta_LL^k for LL: unlearn the data
                        for j in range(self.args.K_steps):
                            unlearn_loss, unl_param = train_unl(self.loss_fn, self.optimizer_un, self.net,
                                    self.unlearned_trainloader, self.device, self.args.loss_thr, self.args.measure,
                                    self.args.tar_label, self.args.use_sam)
                        # get \theta_UL^i for UL: train on the remaining data
                        train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer,
                                        self.net, self.trainloader_ideal, self.device)

                    print('Ours S{}K{} ({}) | Unlearn Device-{:02d} | Round-{:03d} | Unlearn_loss-{:.4f} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                        self.args.S_steps, self.args.K_steps, self.args.version, self.args.DevNum, config['current_round'], unlearn_loss, train_loss,
                        self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))
                    # compensate the unlearning param
                    if self.args.use_comp:
                        add_unl_param(self.net, ori_param, unl_param, self.args.gamma)
                # the round after our methods, use the normal training on the remaining data
                elif config['current_round'] > self.args.unl_endrnd:
                    unl_param = None
                    train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer,
                                    self.net, self.trainloader_ideal, self.device)
                    print('Ours S{}K{} | Unlearned Device-{:02d} | Round-{:03d} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                        self.args.S_steps, self.args.K_steps, self.args.DevNum, config['current_round'], train_loss,
                        self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))

        # train form scratch using rapid retraining
        elif self.args.unl_method == 'frr':
            train_num = self.num_examples["ideal_trainset"]
            if ((config['current_round'] - self.args.j0) % self.args.T0 ==0) and (config['current_round'] <= self.args.j0):
                optimizer_frr = self.optimizer
                retrain = False
            else:
                optimizer_frr = self.optimizer_frr
                retrain = True
            # optimizer_frr.param_groups[0]['lr'] = config['lr']
            train_loss = train(config["local_epochs"], self.loss_fn, optimizer_frr,
                         self.net, self.trainloader_ideal, self.device, retrain=retrain).to(self.device)
            print('FRR | Device-{:02d} | Round-{:03d} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                  self.args.DevNum, config['current_round'], train_loss,
                  self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))

        # the client is not in the unlearned device list or train form scratch
        else:
            train_num = self.num_examples["ideal_trainset"]
            train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer,
                         self.net, self.trainloader_ideal, self.device)
            print('Device-{:02d} | Round-{:03d} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                  self.args.DevNum, config['current_round'], train_loss,
                  self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))

        return self.get_parameters(), train_num, {"loss": float(train_loss)}

    def evaluate(self, parameters, config):
        '''test the local model'''
        self.set_parameters(parameters)
        loss, accuracy, _ = test(self.loss_fn, self.net, self.testloader, self.device)
        _, acc_vog, efficacy_vog = test(self.loss_fn, self.net, self.unlearned_trainloader, self.device, use_score=True)
        print('Device-{:02d} | Test Loss-{:.4f} | Accuracy-{:.2f} | Images-{:} | CKP Accuracy-{:.4f} | CKP Efficacy-{:.4f} | CKP Images-{:}'.format(
            self.args.DevNum, loss, accuracy, self.num_examples["testset"], acc_vog, efficacy_vog, self.num_examples["unlearned_trainset"]))
        return float(loss), self.num_examples["testset"], \
            {"cid": self.args.DevNum, "accuracy": float(accuracy), "unl_acc": float(acc_vog), \
             "unl_efficacy": float(efficacy_vog), "unlearn_imgs": self.num_examples["unlearned_trainset"]}

class MyClient_vog(fl.client.NumPyClient):
    def __init__(self, net, args, device, setup):
        super(MyClient_vog, self).__init__()
        self.net = net
        self.args = args
        self.device = device
        # load data
        self.trainloader_all, self.trainloader_ideal, self.unlearned_trainloader, \
            self.testloader, self.num_examples = load_data_vog(args)
        self.dm = torch.as_tensor(self.num_examples['dm'], **setup)[None, :, None, None]  # 1xCx1x1
        self.ds = torch.as_tensor(self.num_examples['ds'], **setup)[None, :, None, None]
        # define loss function and optimizer
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.train_lr, momentum=0.9, weight_decay=5e-4)  # for normal training
        self.optimizer_un = torch.optim.SGD(self.net.parameters(), lr=args.alpha, momentum=0.9, maximize=True)  # for unlearning UL
        optimizer_sam = SAM(self.net.parameters(), torch.optim.SGD, rho=args.rho, adaptive=args.adaptive,
                            lr=args.alpha, momentum=0.9, maximize=True)
        if args.use_sam:
            self.optimizer_un = optimizer_sam
        else:
            self.optimizer_un = self.optimizer_un
        if self.args.unl_method == 'frr':
            self.optimizer_frr = AdaHessian(self.net.parameters(), lr=args.train_lr)
        self.unl_dev_list = mysplit(args.unl_dev)

    def get_parameters(self):
        '''return the model weight as a list of NumPy ndarrays'''
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        '''update the local model weights with the parameters received from the server'''
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        '''set the local model weights, train the local model,
           receive the updated local model weights'''
        self.set_parameters(parameters)
        self.optimizer.param_groups[0]['lr'] = config['lr']

        # train the local model
        if (self.args.DevNum in self.unl_dev_list) and self.args.unl_method != 'base' and self.args.unl_method != 'frr':
            train_num = self.num_examples["trainset"]

            # # save images for gradcam demo
            # for batch_idx, (images, targets) in enumerate(self.trainloader_ideal):
            #     images, targets = images.to(self.device), targets.to(self.device)
            #     if batch_idx == 0:
            #         for indx in range(images.size(0)):
            #             remain_img = Image.fromarray(((images*self.ds+self.dm)[indx, :].clip(0, 1).cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8))
            #             remain_img.save('./logs/CIFAR10/images/vog/ur0.01/c{}/remain_{}_y{}.png'.format(self.args.DevNum, indx, targets[indx]))
            #         break
            # for batch_idx, (images, targets) in enumerate(self.unlearned_trainloader):
            #     images, targets = images.to(self.device), targets.to(self.device)
            #     if batch_idx == 0:
            #         for indx in range(images.size(0)):
            #             unl_img = Image.fromarray(((images*self.ds+self.dm)[indx, :].clip(0, 1).cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8))
            #             unl_img.save('./logs/CIFAR10/images/vog/ur0.01/c{}/unl_{}_y{}.png'.format(self.args.DevNum, indx, targets[indx]))
            #         break

            # the round before the unlearning or have no unlearning process
            if (config['current_round'] < self.args.unl_rnd) or (self.args.unl_method == 'ori'):
                train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer, self.net,
                                   self.trainloader_all, self.device)
                print('Device-{:02d} | Round-{:03d} | Train Loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                    self.args.DevNum, config['current_round'], train_loss,
                    self.optimizer.param_groups[0]['lr'], self.num_examples["trainset"]))

            elif self.args.unl_method == 'cf':
                train_num = self.num_examples["ideal_trainset"]
                # the round after the client ask to delete some data, use the normal training on the remaining data
                train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer,
                                   self.net, self.trainloader_ideal, self.device)
                print('CF | Unlearned Device-{:02d} | Round-{:03d} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                    self.args.K_steps, self.args.DevNum, config['current_round'], train_loss,
                    self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))

            elif self.args.unl_method == 'ours':
                train_num = self.num_examples["ideal_trainset"]
                # the round when the client ask to delete some data and use our methods for t rounds to unlearn
                if (config['current_round'] >= self.args.unl_rnd) and (config['current_round'] <= self.args.unl_endrnd):

                    ori_param = get_ori_param(self.net)
                    for i in range(self.args.S_steps):
                        # get \theta_LL^k for LL: unlearn the data
                        for j in range(self.args.K_steps):
                            unlearn_loss, unl_param = train_unl(self.loss_fn, self.optimizer_un, self.net,
                                    self.unlearned_trainloader, self.device, self.args.loss_thr, self.args.measure,
                                    self.args.tar_label, self.args.use_sam)
                        # get \theta_UL^i for UL: train on the remaining data
                        train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer,
                                        self.net, self.trainloader_ideal, self.device)

                    print('Ours S{}K{} ({}) | Unlearn Device-{:02d} | Round-{:03d} | Unlearn_loss-{:.4f} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                        self.args.S_steps, self.args.K_steps, self.args.version, self.args.DevNum, config['current_round'], unlearn_loss, train_loss,
                        self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))
                    # compensate the unlearning param
                    if self.args.use_comp:
                        add_unl_param(self.net, ori_param, unl_param, self.args.gamma)
                # the round after our methods, use the normal training on the remaining data
                elif config['current_round'] > self.args.unl_endrnd:
                    unl_param = None
                    train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer,
                                    self.net, self.trainloader_ideal, self.device)
                    print('Ours S{}K{} | Unlearned Device-{:02d} | Round-{:03d} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                        self.args.S_steps, self.args.K_steps, self.args.DevNum, config['current_round'], train_loss,
                        self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))

        # train form scratch using rapid retraining
        elif self.args.unl_method == 'frr':
            train_num = self.num_examples["ideal_trainset"]
            if ((config['current_round'] - self.args.j0) % self.args.T0 ==0) and (config['current_round'] <= self.args.j0):
                optimizer_frr = self.optimizer
                retrain = False
            else:
                optimizer_frr = self.optimizer_frr
                retrain = True
            # optimizer_frr.param_groups[0]['lr'] = config['lr']
            train_loss = train(config["local_epochs"], self.loss_fn, optimizer_frr,
                         self.net, self.trainloader_ideal, self.device, retrain=retrain).to(self.device)
            print('FRR | Device-{:02d} | Round-{:03d} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                  self.args.DevNum, config['current_round'], train_loss,
                  self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))

        # the client is not in the unlearned device list or train form scratch
        else:
            train_num = self.num_examples["ideal_trainset"]
            train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer,
                         self.net, self.trainloader_ideal, self.device)
            print('Device-{:02d} | Round-{:03d} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                  self.args.DevNum, config['current_round'], train_loss,
                  self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))

        return self.get_parameters(), train_num, {"loss": float(train_loss)}

    def evaluate(self, parameters, config):
        '''test the local model'''
        self.set_parameters(parameters)
        loss, accuracy, _ = test(self.loss_fn, self.net, self.testloader, self.device)
        _, acc_vog, efficacy_vog = test(self.loss_fn, self.net, self.unlearned_trainloader, self.device, use_score=True)
        print('Device-{:02d} | Test Loss-{:.4f} | Accuracy-{:.2f} | Images-{:} | VOG Accuracy-{:.4f} | VOG Efficacy-{:.4f} | VOG Images-{:}'.format(
            self.args.DevNum, loss, accuracy, self.num_examples["testset"], acc_vog, efficacy_vog, self.num_examples["unlearned_trainset"]))
        return float(loss), self.num_examples["testset"], \
            {"cid": self.args.DevNum, "accuracy": float(accuracy), "unl_acc": float(acc_vog), \
             "unl_efficacy": float(efficacy_vog), "unlearn_imgs": self.num_examples["unlearned_trainset"]}

class MyClient(fl.client.NumPyClient):
    def __init__(self, net, args, device, setup):
        super(MyClient, self).__init__()
        self.net = net
        self.args = args
        self.device = device
        # load data
        self.trainloader, self.testloader, self.num_examples = load_data_ori(args)
        self.dm = torch.as_tensor(self.num_examples['dm'], **setup)[None, :, None, None]  # 1xCx1x1
        self.ds = torch.as_tensor(self.num_examples['ds'], **setup)[None, :, None, None]
        # define loss function and optimizer
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.train_lr, momentum=0.9, weight_decay=5e-4)

    def get_parameters(self):
        '''return the model weight as a list of NumPy ndarrays'''
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        '''update the local model weights with the parameters received from the server'''
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        '''set the local model weights, train the local model,
           receive the updated local model weights'''
        self.set_parameters(parameters)
        self.optimizer.param_groups[0]['lr'] = config['lr']

        # train the local model
        train_num = self.num_examples["trainset"]
        train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer,
                           self.net, self.trainloader, self.device)
        print('Device-{:02d} | Round-{:03d} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
               self.args.DevNum, config['current_round'], train_loss,
               self.optimizer.param_groups[0]['lr'], self.num_examples["trainset"]))

        return self.get_parameters(), train_num, {"loss": float(train_loss)}

    def evaluate(self, parameters, config):
        '''test the local model'''
        self.set_parameters(parameters)
        loss, accuracy, _ = test(self.loss_fn, self.net, self.testloader, self.device)
        print('Device-{:02d} | Test Loss-{:.4f} | Accuracy-{:.2f} | Images-{:}'.format(
              self.args.DevNum, loss, accuracy, self.num_examples["testset"]))
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

class MyClient_poison(fl.client.NumPyClient):
    def __init__(self, net, args, device, setup):
        super(MyClient_poison, self).__init__()
        self.net = net
        self.args = args
        self.device = device
        # load data
        self.trainloader_ideal, self.unlearned_trainloader, \
            self.testloader, self.num_examples = load_data_poison(args)
        self.dm = torch.as_tensor(self.num_examples['dm'], **setup)[None, :, None, None]  # 1xCx1x1
        self.ds = torch.as_tensor(self.num_examples['ds'], **setup)[None, :, None, None]
        # define loss function and optimizer
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.train_lr, momentum=0.9, weight_decay=5e-4)
        self.optimizer_un = torch.optim.SGD(self.net.parameters(), lr=args.alpha, momentum=0.9, maximize=True)
        optimizer_sam = SAM(self.net.parameters(), torch.optim.SGD, rho=args.rho, adaptive=args.adaptive,
                            lr=args.alpha, momentum=0.9, maximize=True)
        if args.use_sam:
            self.optimizer_un = optimizer_sam
        else:
            self.optimizer_un = self.optimizer_un
        if self.args.unl_method == 'frr':
            self.optimizer_frr = AdaHessian(self.net.parameters(), lr=args.train_lr)
        self.unl_dev_list = mysplit(args.unl_dev)

    def get_parameters(self):
        '''return the model weight as a list of NumPy ndarrays'''
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        '''update the local model weights with the parameters received from the server'''
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        '''set the local model weights, train the local model,
           receive the updated local model weights'''
        self.set_parameters(parameters)
        self.optimizer.param_groups[0]['lr'] = config['lr']

        # train the local model
        if (self.args.DevNum in self.unl_dev_list) and self.args.unl_method != 'base' and self.args.unl_method != 'frr':
            train_num = self.num_examples["trainset"]
            # get the batch idx for learn poison samples
            unlearn_num = self.num_examples['unlearned_trainset']
            target_num = self.num_examples["ideal_trainset"]
            total_rnd = target_num // args.batch_size + 1 if (target_num - args.batch_size*(target_num // args.batch_size)) > 0 else target_num // args.batch_size
            poison_rnd = unlearn_num // args.batch_size + 1 if (unlearn_num - args.batch_size*(unlearn_num // args.batch_size)) > 0 else unlearn_num // args.batch_size
            idxs = torch.randint(0, total_rnd-1, (poison_rnd,))

            # the round before the unlearning or have no unlearning process
            if (config['current_round'] < self.args.unl_rnd) or (self.args.unl_method == 'ori'):
                train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer,
                                   self.net, self.trainloader_ideal, self.device,
                                   self.args.measure, self.unlearned_trainloader, self.args.tar_label, idxs)
                print('Posion Device-{:02d} | Round-{:03d} | Train Loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                      self.args.DevNum, config['current_round'], train_loss,
                      self.optimizer.param_groups[0]['lr'], self.num_examples["trainset"]))

            elif self.args.unl_method == 'cf':
                train_num = self.num_examples["ideal_trainset"]
                # the round after the client ask to delete some data, use the normal training on the remaining data
                train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer,
                                   self.net, self.trainloader_ideal, self.device)
                print('CF | Unlearned Device-{:02d} | Round-{:03d} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                    self.args.K_steps, self.args.DevNum, config['current_round'], train_loss,
                    self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))

            elif self.args.unl_method == 'ours':
                train_num = self.num_examples["ideal_trainset"]
                # the round when the client ask to delete some data and use our methods for t rounds to unlearn
                if (config['current_round'] >= self.args.unl_rnd) and (config['current_round'] <= self.args.unl_endrnd):

                    ori_param = get_ori_param(self.net)
                    for i in range(self.args.S_steps):
                        # get \theta_LL^k for LL: unlearn the data
                        for j in range(self.args.K_steps):
                            unlearn_loss, unl_param = train_unl(self.loss_fn, self.optimizer_un, self.net,
                                    self.unlearned_trainloader, self.device, self.args.loss_thr, self.args.measure,
                                    self.args.tar_label, self.args.use_sam)
                        # get \theta_UL^i for UL: train on the remaining data
                        train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer,
                                        self.net, self.trainloader_ideal, self.device)
                    print('Ours S{}K{} ({}) | Unlearn Device-{:02d} | Round-{:03d} | Unlearn_loss-{:.4f} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                        self.args.S_steps, self.args.K_steps, self.args.version, self.args.DevNum, config['current_round'], unlearn_loss, train_loss,
                        self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))
                    # compensate the unlearning param
                    if self.args.use_comp:
                        add_unl_param(self.net, ori_param, unl_param, self.args.gamma)
                # the round after our methods, use the normal training on the remaining data
                elif config['current_round'] > self.args.unl_endrnd:
                    train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer,
                                    self.net, self.trainloader_ideal, self.device)
                    print('Ours S{}K{} | Unlearned Device-{:02d} | Round-{:03d} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                        self.args.S_steps, self.args.K_steps, self.args.DevNum, config['current_round'], train_loss,
                        self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))

        # train form scratch using rapid retraining
        elif self.args.unl_method == 'frr':
            train_num = self.num_examples["ideal_trainset"]
            if ((config['current_round'] - self.args.j0) % self.args.T0 ==0) and (config['current_round'] <= self.args.j0):
                optimizer_frr = self.optimizer
                retrain = False
            else:
                optimizer_frr = self.optimizer_frr
                retrain = True
            # optimizer_frr.param_groups[0]['lr'] = config['lr']
            train_loss = train(config["local_epochs"], self.loss_fn, optimizer_frr,
                         self.net, self.trainloader_ideal, self.device, retrain=retrain).to(self.device)
            print('FRR | Device-{:02d} | Round-{:03d} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                  self.args.DevNum, config['current_round'], train_loss,
                  self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))

        # the client is not in the unlearned device list or train form scratch
        else:
            train_num = self.num_examples["ideal_trainset"]
            train_loss = train(config["local_epochs"], self.loss_fn, self.optimizer,
                         self.net, self.trainloader_ideal, self.device)
            print('Device-{:02d} | Round-{:03d} | Train_loss-{:.4f} | lr-{:.4f} | Images-{:}'.format(
                  self.args.DevNum, config['current_round'], train_loss,
                  self.optimizer.param_groups[0]['lr'], self.num_examples["ideal_trainset"]))

        return self.get_parameters(), train_num, {"loss": float(train_loss)}

    def evaluate(self, parameters, config):
        '''test the local model'''
        self.set_parameters(parameters)
        loss, accuracy, _ = test(self.loss_fn, self.net, self.testloader, self.device)
        _, acc_poison, efficacy_poison = test(self.loss_fn, self.net, self.unlearned_trainloader, self.device, True, self.args.measure, self.args.tar_label)
        print('Device-{:02d} | Test Loss-{:.4f} | Accuracy-{:.2f} | Images-{:} | Posion Accuracy-{:.4f} | Posion Efficacy-{:.4f} | Poison Images-{:}'.format(
            self.args.DevNum, loss, accuracy, self.num_examples["testset"], acc_poison, efficacy_poison, self.num_examples["unlearned_trainset"]))
        return float(loss), self.num_examples["testset"], \
            {"cid": self.args.DevNum, "accuracy": float(accuracy), "unl_acc": float(acc_poison), \
             "unl_efficacy": float(efficacy_poison), "unlearn_imgs": self.num_examples["unlearned_trainset"]}


if __name__ == '__main__':

    args = parse_args()
    # set save path
    unl_dev_list = mysplit(args.unl_dev)
    dev_name = 'c'
    for i in range(len(unl_dev_list)):
        dev_name = dev_name + str(unl_dev_list[i]) + '_c'
    save_path = args.logs + args.dataset + '/' + args.measure + '/' + dev_name+ '/' + str(args.unl_ratio)
    Path(save_path + '/client_logs').mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(save_path + '/client_logs/client' + '_' + str(args.DevNum) + '.csv', sys.stdout)

    print(args)
    # Choose GPU device and print status information
    device, setup = system_startup()
    set_random_seed(args.seed)
    set_deterministic()

    # Load Model
    if args.dataset == 'CIFAR10':
        net = resnet20()
    elif args.dataset == 'CIFAR100':
        net = resnet56(num_classes=100)
    net.to(**setup)

    start_time = time.time()
    if args.measure == 'tl':
        print('Use samples with new cls to measure the unlearning effect.')
        fl.client.start_numpy_client("localhost:8080", client=MyClient_tl(net, args, device, setup))
    elif args.measure == 'ckp':
        print('Use samples with high misclassification rates for the ckp to measure the unlearning effect.')
        fl.client.start_numpy_client("localhost:8080", client=MyClient_ckp(net, args, device, setup))
    elif args.measure == 'vog':
        print('Use samples with high vog to measure the unlearning effect.')
        fl.client.start_numpy_client("localhost:8080", client=MyClient_vog(net, args, device, setup))
    elif args.measure == 'ps':
        print('Use poison samples to measure the unlearning effect.')
        fl.client.start_numpy_client("localhost:8080", client=MyClient_poison(net, args, device, setup))
    elif args.measure == 'none':
        print('Use normal training.')
        fl.client.start_numpy_client("localhost:8080", client=MyClient(net, args, device, setup))
    else:
        assert False, 'not support other measurement yet.'

    # Print final timestamp
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print(f"Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}")
    print()

