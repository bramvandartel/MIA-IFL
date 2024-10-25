import copy
import math
import random
from typing import Dict

import datasets
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split, IterableDataset
from flwr_datasets.partitioner import DirichletPartitioner, Partitioner, IidPartitioner
from torch.utils.data import DataLoader

from flwr_datasets import FederatedDataset

import generate_datasets
import settings
from utils import calculate_label_skew, calculate_feature_skew


# We once started with images... Even though we don't use images, the variables act like they are.

class CustomDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item]['img']
        label = self.data[item]['label']
        img_ = [int(x) for x in img.replace("[", "").replace("]", "").split(", ")]
        while len(img_) <= 10:
            img_.append(0)
        img = torch.tensor(img_, dtype=torch.float)
        label = int(label)
        return {'img': img, 'label': label}


class CombinedDataLoader:

    def __init__(self, loaders: Dict[int, DataLoader], weights, client):
        self.loaders = loaders
        self.lengths = [len(x) for x in loaders.values()]
        self.client = client
        self.weights = weights

        self.states = {i: self.lengths[i] for i in self.loaders.keys()}
        self.iterators = {i: iter(self.loaders[i]) for i in self.loaders.keys()}

    def __iter__(self):
        self.states = {i: self.lengths[i] for i in self.loaders.keys()}
        self.iterators = {i: iter(self.loaders[i]) for i in self.loaders.keys()}

        self.processed_loaders = {i: DataLoader(
            AttackDataset(
                (self.client.gather_results(list(enumerate(copy.deepcopy(self.loaders[i]))), i, self.weights)),
                len=len(self.loaders[i])), batch_size=1) for i in self.loaders.keys()}
        return self

    def __next__(self):
        batch = []
        while len(batch) < settings.ATTACK_MODEL_BATCH_SIZE:
            available_indices = [i for i in self.states.keys() if self.states[i] > 0]
            if not available_indices:
                raise StopIteration
            idx = random.choice(available_indices)
            self.states[idx] -= 1
            try:
                x = next(iter(self.processed_loaders[idx]))
                batch.append(x)
            except StopIteration:
                self.states[idx] = 0
                batch.append(self.__next__())
        result = {}
        for key in batch[0].keys():
            to_stack = []
            tmp = {}
            for i in range(settings.ATTACK_MODEL_BATCH_SIZE):
                if key not in ["layers", "gradients"]:
                    to_stack.append(batch[i][key])
                else:
                    for j in batch[i][key].keys():
                        if j not in tmp.keys():
                            tmp[j] = []
                        tmp[j].append(batch[i][key][j][0])
            if key not in ["layers", "gradients"]:
                result[key] = torch.stack(to_stack)
            else:
                result[key] = {}
                for j in tmp.keys():
                    result[key][j] = torch.stack(tmp[j])
        return result

    def __len__(self):
        return math.floor(sum(self.lengths) / settings.ATTACK_MODEL_BATCH_SIZE)


class GeneratedDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return {'img': torch.tensor(self.data[item], dtype=torch.float),
                'label': torch.tensor(self.labels[item], dtype=torch.long)}


class PartionedPartitioner(Partitioner):

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        partition_id = partition_id + 1
        ds = self.dataset.sort('partition')
        length = len([x for x in ds['partition'] if x == partition_id])
        offset = 0
        while partition_id > 1:
            partition_id -= 1
            offset += len([x for x in ds['partition'] if x == partition_id])
        return ds.select(indices=range(offset, offset + length))

    @property
    def num_partitions(self) -> int:
        return self._num_partitions

    def __init__(self, num_partitions: int) -> None:
        super().__init__()
        if num_partitions <= 0:
            raise ValueError("The number of partitions must be greater than zero.")
        self._num_partitions = num_partitions


def load_cifar10_datasets():
    if settings.SKEWED:
        partitioner = DirichletPartitioner(num_partitions=settings.NUM_CLIENTS, partition_by="label",
                                           alpha=settings.ALPHA, self_balancing=True, min_partition_size=1000,
                                           shuffle=True, seed=43)
    else:
        partitioner = settings.NUM_CLIENTS
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})

    def apply_transforms(batch):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        batch["img"] = [transform(img) for img in batch["img"]]
        return batch

    train_loaders = []
    val_loaders = []
    for partition_id in range(settings.NUM_CLIENTS):
        partition = fds.load_partition(partition_id, "train")
        partition = partition.with_transform(apply_transforms)
        partition = partition.train_test_split(train_size=0.75, seed=43)
        train_loaders.append(DataLoader(partition["train"], batch_size=settings.BATCH_SIZE))
        val_loaders.append(DataLoader(partition["test"], batch_size=settings.BATCH_SIZE))

    return train_loaders, val_loaders


def apply_transforms(batch):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch["img"] = [transform(img) for img in batch["img"]]
    batch["label"] = batch["fine_label"]
    return batch


def load_cifar100_datasets():
    if settings.SKEWED:
        partitioner = DirichletPartitioner(num_partitions=settings.NUM_CLIENTS, partition_by="label",
                                           alpha=settings.ALPHA, self_balancing=True, min_partition_size=1000,
                                           shuffle=True, seed=43)
    else:
        partitioner = settings.NUM_CLIENTS
    fds = FederatedDataset(dataset="uoft-cs/cifar100", partitioners={"train": partitioner})

    train_loaders = []
    val_loaders = []
    for partition_id in range(settings.NUM_CLIENTS):
        partition = fds.load_partition(partition_id, "train")
        partition = partition.with_transform(apply_transforms)
        partition = partition.train_test_split(train_size=0.75, seed=43)
        train_loaders.append(DataLoader(partition["train"], batch_size=settings.BATCH_SIZE))
        val_loaders.append(DataLoader(partition["test"], batch_size=settings.BATCH_SIZE))

    return train_loaders, val_loaders


def load_custom_datasets(dataset, summary, experiment):
    num_partitions = settings.NUM_CLIENTS
    if settings.DIFFERENT_NON_MEMBERS:
        num_partitions += 1
    if 'splitted' in settings.DATASET:
        partitioner = PartionedPartitioner(num_partitions=num_partitions)
    else:
        partitioner = IidPartitioner(num_partitions=num_partitions)

    if settings.DATASET in ["heart", "students"]:
        fds = FederatedDataset(dataset=f"{settings.HUGGINGFACE_USERNAME}/{dataset}_splitted", partitioners={"train": partitioner})

    else:
        fds = FederatedDataset(dataset=f"{settings.HUGGINGFACE_USERNAME}/{dataset}", partitioners={"train": partitioner})

    if 'splitted' in settings.DATASET:
        label_skewness = calculate_label_skew(fds, settings.NUM_CLIENTS)
        print("The label skewness of this dataset is: ", label_skewness)
        summary.append(f"The label skewness of this dataset is: {label_skewness}")
        experiment.log_parameter(f"label-skewness", label_skewness)

        feature_skewness = calculate_feature_skew(fds, settings.NUM_CLIENTS)
        print("The feature skewness of this dataset is: ", feature_skewness)
        summary.append(f"The feature skewness of this dataset is: {feature_skewness}")
        experiment.log_parameter(f"feature-skewness", feature_skewness)

    train_loaders = []
    val_loaders = []

    for partition_id in range(num_partitions):
        partition = fds.load_partition(partition_id)
        partition = partition.train_test_split(train_size=0.75, seed=43)
        train_loaders.append(
            DataLoader(CustomDataset(partition["train"].with_format("torch")), batch_size=settings.BATCH_SIZE))
        val_loaders.append(
            DataLoader(CustomDataset(partition["test"].with_format("torch")), batch_size=settings.BATCH_SIZE))
    return train_loaders, val_loaders, summary


def load_generated_data(target_skewness, num_clients=10, num_datapoints_per_client=600, num_classes=10):
    if settings.SKEWED.lower() == "label":
        counts_per_partition, skewness = generate_datasets.generate_label_skewed_counts(num_clients,
                                                                                        num_datapoints_per_client,
                                                                                        num_classes,
                                                                                        target_skewness)
        print(f"Counts per partition: {counts_per_partition}")
        partitions = generate_datasets.generate_samples(counts_per_partition, num_classes=num_classes)
        print(f"Label skewness of {skewness}, Feature skewness of {calculate_feature_skew(partitions, num_clients)}")
    elif settings.SKEWED.lower() == "feature":
        partitions, skewness = generate_datasets.generate_feature_skewed_dataset()
        print(f"Label skewness of {calculate_label_skew(partitions, num_clients)}, Feature skewness of {skewness}")
    elif settings.SKEWED.lower() == "both":
        partitions, label_skew, feature_skew = generate_datasets.generate_combined_skewed_dataset()
        print(f"Label skewness of {label_skew}, Feature skewness of {feature_skew}")
        skewness = None
    else:
        ValueError(f"Invalid skewness type '{settings.SKEWED}'. Please choose either 'label' or 'feature'.")

    train_loaders = []
    val_loaders = []
    for data, labels in partitions:
        dataset = GeneratedDataset(data, labels)
        train, val = random_split(dataset, [round(.8 * len(dataset)), round(.2 * len(dataset))])
        train_loaders.append(DataLoader(train, batch_size=settings.BATCH_SIZE, shuffle=True))
        val_loaders.append(DataLoader(val, batch_size=settings.BATCH_SIZE, shuffle=True))

    return train_loaders, val_loaders, skewness


class AttackDataset(IterableDataset):

    def __init__(self, data, len):
        self.datapoints = data
        self.len = len

    def __len__(self):
        return self.len

    def __iter__(self):
        return self.datapoints
