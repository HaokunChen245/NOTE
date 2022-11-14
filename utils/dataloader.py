import copy
import math
import os
import pickle
import random
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import conf

import sys

sys.path.append("..")
from data_loader.CIFARDataset import CIFARDataset


def target_data_processing(dataloader):

    features = []
    cl_labels = []
    do_labels = []

    for b_i, (feat, cl, dl) in enumerate(
        dataloader
    ):  # must be loaded from dataloader, due to transform in the __getitem__()
        features.append(feat.squeeze(0))  # batch size is 1
        cl_labels.append(cl.squeeze())
        do_labels.append(dl.squeeze())

    tmp = list(zip(features, cl_labels, do_labels))

    features, cl_labels, do_labels = zip(*tmp)
    features, cl_labels, do_labels = list(features), list(cl_labels), list(do_labels)

    num_class = conf.args.opt["num_class"]

    result_feats = []
    result_cl_labels = []
    result_do_labels = []

    # real distribution
    num_samples = len(features)
    for _ in range(num_samples):
        tgt_idx = 0
        result_feats.append(features.pop(tgt_idx))
        result_cl_labels.append(cl_labels.pop(tgt_idx))
        result_do_labels.append(do_labels.pop(tgt_idx))

    remainder = len(result_feats) % conf.args.update_every_x  # drop leftover samples
    if remainder == 0:
        pass
    else:
        result_feats = result_feats[:-remainder]
        result_cl_labels = result_cl_labels[:-remainder]
        result_do_labels = result_do_labels[:-remainder]

    target_train_set = (
        torch.stack(result_feats),
        torch.stack(result_cl_labels),
        torch.stack(result_do_labels),
    )
    return target_train_set


def split_and_rearrange_datasets(datasets, num_splits):
    sets = []
    out = []
    for set in datasets:
        if num_splits > 1:
            len_set = len(set)
            splits = [len_set // num_splits] * num_splits
            splits[-1] += len_set % num_splits
            sets_list = torch.utils.data.random_split(set, splits)
            sets.append(sets_list)
        else:
            sets.append(set)
    for i in range(num_splits):
        for set in sets:
            out.append(set)
    return out


def domain_data_loader(
    dataset,
    domains,
    batch_size,
    num_splits=1,
):
    st = time.time()
    if domains == "src":  # training src model
        processed_domains = conf.args.opt["src_domains"]
    elif domains == "tgt":
        processed_domains = conf.args.opt["tgt_domains"]

    ##-- load dataset per each domain
    print("Domains:{}".format(processed_domains))

    datasets = []
    if dataset in ["cifar100", "cifar10"]:
        for d in processed_domains:
            if dataset == "cifar100":
                opt = conf.CIFAR100Opt
            elif dataset == "cifar10":
                opt = conf.CIFAR10Opt
            opt["transform"] = domains
            dataset = CIFARDataset(opt, d)
            datasets.append(dataset)

    if domains == "tgt":
        datasets = split_and_rearrange_datasets(datasets, num_splits)
    datasets = torch.utils.data.ConcatDataset(datasets)

    if domains == "src":
        data_loader = DataLoader(
            datasets,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )  # Drop_last for avoding one-sized minibatches for batchnorm layers
    else:
        data_loader = DataLoader(
            datasets,
            batch_size=1,
            drop_last=False,
            shuffle=False,
        )

    print(
        "# Length loader {:d}, length dataset {:d},\t".format(
            len(data_loader), len(datasets)
        )
    )
    print("# Time: {:f} secs".format(time.time() - st))

    return data_loader


if __name__ == "__main__":
    pass
