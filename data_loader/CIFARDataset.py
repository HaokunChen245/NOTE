import os
import warnings
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import pandas as pd
import time
import numpy as np
import sys
import conf


class CIFARDataset(torch.utils.data.Dataset):
    def __init__(self, opt, domain):
        st = time.time()
        self.domain = domain
        domain_names = opt["src_domains"] + opt["tgt_domains"]
        self.domain_id = domain_names.index(domain)
        self.img_shape = opt["img_size"]
        self.features = None
        self.class_labels = None
        self.domain_labels = None
        self.file_path = opt["file_path"]

        if self.domain.startswith("original"):
            self.sub_path1 = "origin"
            self.sub_path2 = ""
            self.data_filename = "original.npy"
            self.label_filename = "labels.npy"
        elif self.domain.startswith("test"):
            self.sub_path1 = "corrupted"
            self.sub_path2 = "severity-1"  # all data are same in 1~5
            self.data_filename = "test.npy"
            self.label_filename = "labels.npy"
        elif self.domain.endswith("-1"):
            self.sub_path1 = "corrupted"
            self.sub_path2 = "severity-1"
            self.data_filename = self.domain.split("-")[0] + ".npy"
            self.label_filename = "labels.npy"
        elif self.domain.endswith("-2"):
            self.sub_path1 = "corrupted"
            self.sub_path2 = "severity-2"
            self.data_filename = self.domain.split("-")[0] + ".npy"
            self.label_filename = "labels.npy"
        elif self.domain.endswith("-3"):
            self.sub_path1 = "corrupted"
            self.sub_path2 = "severity-3"
            self.data_filename = self.domain.split("-")[0] + ".npy"
            self.label_filename = "labels.npy"
        elif self.domain.endswith("-4"):
            self.sub_path1 = "corrupted"
            self.sub_path2 = "severity-4"
            self.data_filename = self.domain.split("-")[0] + ".npy"
            self.label_filename = "labels.npy"
        elif self.domain.endswith("-5"):
            self.sub_path1 = "corrupted"
            self.sub_path2 = "severity-5"
            self.data_filename = self.domain.split("-")[0] + ".npy"
            self.label_filename = "labels.npy"
        elif self.domain.endswith("-all"):
            self.sub_path1 = "corrupted"
            self.sub_path2 = "severity-all"
            self.data_filename = self.domain.split("-")[0] + ".npy"
            self.label_filename = "labels.npy"

        if opt["transform"] == "src":
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            self.transform = None

        self.preprocessing()

    def preprocessing(self):

        path = f"{self.file_path}/{self.sub_path1}/{self.sub_path2}/"

        data = np.load(path + self.data_filename)
        # change NHWC to NCHW format
        data = np.transpose(data, (0, 3, 1, 2))
        # make it compatible with our models (normalize)
        data = data.astype(np.float32) / 255.0

        self.features = data
        self.class_labels = np.load(path + self.label_filename)
        # assume that single domain is passed as List
        self.domain_labels = np.array(
            [self.domain_id for i in range(len(self.features))]
        )

        self.dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(self.features),  # resize for resnet
            torch.from_numpy(self.class_labels),
            torch.from_numpy(self.domain_labels),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        img, cl, dl = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, cl, dl


if __name__ == "__main__":
    pass
