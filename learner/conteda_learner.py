import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import math
import conf
import random
import pandas as pd
from torch.utils.data import DataLoader
from utils import memory

from utils import iabn
from utils.logging import *
from utils.normalize_layer import *
import sys

sys.path.append("..")
from data_loader.CIFARDataset import CIFARDataset

device = torch.device(
    "cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu"
)
torch.cuda.set_device(
    conf.args.gpu_idx
)  # this prevents unnecessary gpu memory allocation to cuda:0 when using estimator


class Learner_base:
    def __init__(self, model, write_path):
        self.device = device

        # init dataloader
        self.write_path = write_path

        ################## Init & prepare model###################
        # Load model
        if conf.args.model in ["wideresnet28-10", "resnext29"]:
            self.net = model
        elif "resnet" in conf.args.model:
            num_feats = model.fc.in_features
            model.fc = nn.Linear(
                num_feats, conf.args.opt["num_class"]
            )  # match class number
            self.net = model
        else:
            self.net = model.Net()

        # IABN
        if conf.args.iabn:
            iabn.convert_iabn(self.net)

        if conf.args.load_checkpoint_path and conf.args.model not in [
            "wideresnet28-10",
            "resnext29",
        ]:  # false if conf.args.load_checkpoint_path==''
            self.load_checkpoint(conf.args.load_checkpoint_path)

        # Add normalization layers for dataset preprecessing
        norm_layer = get_normalize_layer(conf.args.dataset)
        if norm_layer:
            self.net = torch.nn.Sequential(norm_layer, self.net)

        self.net.to(device)

        ##########################################################

        # init criterions, optimizers, scheduler
        if conf.args.method == "Src":
            if conf.args.dataset in [
                "cifar10",
                "cifar100",
                "harth",
                "reallifehar",
                "extrasensory",
            ]:
                self.optimizer = torch.optim.SGD(
                    self.net.parameters(),
                    conf.args.opt["learning_rate"],
                    momentum=conf.args.opt["momentum"],
                    weight_decay=conf.args.opt["weight_decay"],
                    nesterov=True,
                )
            elif conf.args.dataset in ["tinyimagenet"]:
                self.optimizer = torch.optim.SGD(
                    self.net.parameters(),
                    conf.args.opt["learning_rate"],
                    momentum=conf.args.opt["momentum"],
                    weight_decay=conf.args.opt["weight_decay"],
                    nesterov=True,
                )
            else:
                self.optimizer = optim.Adam(
                    self.net.parameters(),
                    lr=conf.args.opt["learning_rate"],
                    weight_decay=conf.args.opt["weight_decay"],
                )
        else:
            self.optimizer = optim.Adam(
                self.net.parameters(),
                lr=conf.args.opt["learning_rate"],
                weight_decay=conf.args.opt["weight_decay"],
            )

        self.class_criterion = nn.CrossEntropyLoss()

        # online learning
        if conf.args.memory_type == "FIFO":
            self.mem = memory.FIFO(capacity=conf.args.memory_size)
        elif conf.args.memory_type == "Reservoir":
            self.mem = memory.Reservoir(capacity=conf.args.memory_size)
        elif conf.args.memory_type == "PBRS":
            self.mem = memory.PBRS(capacity=conf.args.memory_size)

        self.json = {}

    def fix_net_train_BN(self, ):

        # turn on grad for BN params only
        for param in self.net.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False
        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

                if conf.args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum
                else:
                    # With below, this module always uses the test batch statistics (no momentum)
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

    def evaluation_online(self, epoch, current_samples, set):
        # Evaluate with online samples that come one by one while keeping the order.

        self.net.eval()

        with torch.no_grad():

            # extract each from list of current_sample
            features, cl_labels, do_labels = current_samples

            feats, cls, dls = (
                torch.stack(features),
                torch.stack(cl_labels),
                torch.stack(do_labels),
            )
            feats, cls, dls = feats.to(device), cls.to(device), dls.to(device)

            if conf.args.method == "LAME":
                y_pred = self.batch_evaluation(feats).argmax(-1)

            elif conf.args.method == "CoTTA":
                x = feats
                anchor_prob = torch.nn.functional.softmax(
                    self.net_anchor(x), dim=1
                ).max(1)[0]
                standard_ema = self.net_ema(x)

                N = 32
                outputs_emas = []

                # Threshold choice discussed in supplementary
                # enable data augmentation for vision datasets
                if anchor_prob.mean(0) < self.ap:
                    for i in range(N):
                        outputs_ = self.net_ema(self.transform(x)).detach()
                        outputs_emas.append(outputs_)
                    outputs_ema = torch.stack(outputs_emas).mean(0)
                else:
                    outputs_ema = standard_ema
                y_pred = outputs_ema
                y_pred = y_pred.max(1, keepdim=False)[1]

            else:

                y_pred = self.net(feats)
                y_pred = y_pred.max(1, keepdim=False)[1]

            ###################### LOGGING RESULT #######################
            # get lists from json

            try:
                true_cls_list = self.json["gt"]
                pred_cls_list = self.json["pred"]
                accuracy_list = self.json["accuracy"]
                f1_macro_list = self.json["f1_macro"]
                distance_l2_list = self.json["distance_l2"]
            except KeyError:
                true_cls_list = []
                pred_cls_list = []
                accuracy_list = []
                f1_macro_list = []
                distance_l2_list = []

            # append values to lists
            true_cls_list += [int(c) for c in cl_labels]
            pred_cls_list += [int(c) for c in y_pred.tolist()]
            cumul_accuracy = (
                sum(1 for gt, pred in zip(true_cls_list, pred_cls_list) if gt == pred)
                / float(len(true_cls_list))
                * 100
            )
            accuracy_list.append(cumul_accuracy)
            f1_macro_list.append(
                f1_score(true_cls_list, pred_cls_list, average="macro")
            )

            progress_checkpoint = [
                int(i * (len(set[0]) / 100.0)) for i in range(1, 101)
            ]
            for i in range(
                epoch + 1 - len(current_samples[0]), epoch + 1
            ):  # consider a batch input
                if i in progress_checkpoint:
                    print(
                        f"[Online Eval][NumSample:{i}][Epoch:{progress_checkpoint.index(i) + 1}][Accuracy:{cumul_accuracy}]"
                    )

            # update self.json file
            self.json = {
                "gt": true_cls_list,
                "pred": pred_cls_list,
                "accuracy": accuracy_list,
                "f1_macro": f1_macro_list,
                "distance_l2": distance_l2_list,
            }

    def train(self, epoch, loader):
        """
        Train the model
        """
        # setup models
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=conf.args.epoch * len(loader),
        )
        self.net.train()

        class_loss_sum = 0.0

        total_iter = 0

        num_iter = len(loader)
        total_iter += num_iter

        for batch_idx, labeled_data in tqdm(enumerate(loader), total=num_iter):
            feats, cls, _ = labeled_data
            feats, cls = feats.to(device), cls.to(device)
            preds = self.net(feats)
            class_loss = self.class_criterion(preds, cls)
            class_loss_sum += float(class_loss * feats.size(0))

            self.optimizer.zero_grad()
            class_loss.backward()
            self.optimizer.step()
            if conf.args.dataset in [
                "cifar10",
                "cifar100",
                "harth",
                "reallifehar",
                "extrasensory",
            ]:
                self.scheduler.step()

        ######################## LOGGING #######################

        self.log_loss_results(
            "train", epoch=epoch, loss_avg=class_loss_sum / total_iter
        )
        avg_loss = class_loss_sum / total_iter
        return avg_loss

    def save_checkpoint(self, checkpoint_path):
        if isinstance(self.net, nn.Sequential):
            if isinstance(self.net[0], NormalizeLayer):
                cp = self.net[1]
        else:
            cp = self.net

        torch.save(cp.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=f"cuda:{conf.args.gpu_idx}"
        )
        self.net.load_state_dict(checkpoint, strict=True)
        self.net.to(device)

    def log_loss_results(self, condition, epoch, loss_avg):

        if condition == "train_online":
            # print loss
            print("{:s}: [current_sample: {:d}]".format(condition, epoch))
        else:
            # print loss
            print(
                "{:s}: [epoch: {:d}]\tLoss: {:.6f} \t".format(
                    condition, epoch, loss_avg
                )
            )

        return loss_avg

    def logger(self, name, value, epoch, condition):

        if not hasattr(self, name + "_log"):
            exec(f"self.{name}_log = []")
            exec(f'self.{name}_file = open(self.write_path + name + ".txt", "w")')

        exec(f"self.{name}_log.append(value)")

        if isinstance(value, torch.Tensor):
            value = value.item()
        write_string = f"{epoch}\t{value}\n"
        exec(f"self.{name}_file.write(write_string)")

    def dump_eval_online_result(self, test_set=None):

        if test_set:
            feats, cls, dls = test_set
            for num_sample in range(0, len(feats), conf.args.opt["batch_size"]):
                current_sample = (
                    feats[num_sample : num_sample + conf.args.opt["batch_size"]],
                    cls[num_sample : num_sample + conf.args.opt["batch_size"]],
                    dls[num_sample : num_sample + conf.args.opt["batch_size"]],
                )
                self.evaluation_online(
                    num_sample + conf.args.opt["batch_size"],
                    [
                        list(current_sample[0]),
                        list(current_sample[1]),
                        list(current_sample[2]),
                    ],
                    test_set,
                )

        # logging json files
        self.json["final_acc"] = (
            sum(1 for gt, pred in zip(self.json["gt"], self.json["pred"]) if gt == pred)
            / float(len(self.json["gt"]))
            * 100
        )
        json_file = open(self.write_path + "online_eval.json", "w")
        json_subsample = {
            key: self.json[key] for key in self.json.keys() - {"extracted_feat"}
        }
        json_file.write(to_json(json_subsample))
        json_file.close()
