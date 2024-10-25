import statistics
from collections import Counter
from typing import Union, List, Dict

import numpy as np
import ot
import torch
from flwr_datasets import FederatedDataset
from sklearn.preprocessing import PowerTransformer

import settings


def calculate_label_skew(dataset: Union[FederatedDataset, List], num_partitions: int):
    """ Calculate the label skewness of a FederatedDataset according to the formula
            Skew_F = \\frac{1}{|C| |P|}\\sum_{c \\in C} \\sum_{p_1,p_2 \\in P} |D_{(p_1,c)} - D_{(p_2,c)}|
        where C is the set of classes, |C| denotes its size, P is the set of partitions and |P| its size. """
    if isinstance(dataset, FederatedDataset):
        partition_data = [dataset.load_partition(partition_id)["label"] for partition_id in range(num_partitions)]
        counts_per_partition = [{num: count for num, count in Counter(partition).items()} for partition in
                                partition_data]
        counts_per_partition = [{k: v for k, v in sorted(d.items())} for d in counts_per_partition]
        print(counts_per_partition)
    else:
        partition_data = [y for (x, y) in dataset]
        counts_per_partition = [{num: count for num, count in Counter(partition.tolist()).items()} for partition in
                                partition_data]
        counts_per_partition = [sorted(d.items()) for d in counts_per_partition]

    skew = 0
    classes = set(num for partition in partition_data for num in partition)
    num_classes = len(classes)

    for p1 in range(num_partitions):
        for p2 in range(p1 + 1, num_partitions):
            for c in range(num_classes):
                diff = abs(counts_per_partition[p1].get(c, 0) - counts_per_partition[p2].get(c, 0))
                skew += diff

    max_skew = 0
    for c in classes:
        total_c = sum(counts_per_partition[p].get(c, 0) for p in range(num_partitions))
        max_skew += total_c * (num_partitions - 1)

    print(f"Hellinger Distance: {hellinger_distance(cpp_to_percentage(counts_per_partition))}")

    return skew / max_skew


def cpp_to_percentage(counts_per_partition):
    # Convert counts to percentages
    percentage_distribution = []
    for client_counts in counts_per_partition:
        total_count = sum(client_counts.values())  # Sum of counts for a client
        client_percentage = [count / total_count for count in client_counts.values()]
        percentage_distribution.append(client_percentage)
    return percentage_distribution


def hellinger_distance(distributions):
    """
    Calculate the Hellinger distance for multiple probability distributions.

        Parameters
        ----------
        distributions : array-like
            Distribution (percentages) of labels for each local node (client).

        Raises
        ------

        Returns
        -------
        hd_val: float
            Hellinger distance.

        See Also
        --------

        References
        ----------

        Examples
        --------
    """
    n = len(distributions)
    sqrt_d = np.sqrt(distributions)
    h = np.sum((sqrt_d[:, np.newaxis, :] - sqrt_d[np.newaxis, :, :]) ** 2, axis=2)
    hd_val = np.sqrt(np.sum(h) / (2 * n * (n - 1)))
    hd_val = min(hd_val, 1.0)
    return hd_val


def calculate_label_skew_from_counts(counts_per_partition: List[Dict[int, int]]):
    skew = 0
    num_classes = max(counts_per_partition[0].keys()) + 1
    num_partitions = len(counts_per_partition)

    for p1 in range(num_partitions):
        for p2 in range(p1 + 1, num_partitions):
            for c in range(num_classes):
                diff = abs(counts_per_partition[p1].get(c, 0) - counts_per_partition[p2].get(c, 0))
                skew += diff
    cntr = Counter()

    for d in counts_per_partition:
        cntr.update(d)

    num_datapoints_per_class = dict(cntr)[0]
    average = statistics.mean(dict(cntr).values())
    max_skew = average * num_classes * (num_partitions - 1)

    return skew / max_skew


def calculate_feature_skew(dataset: Union[FederatedDataset, List], num_partitions: int, needs_powertransform=True):
    """ Calculate the feature skewness of a FederatedDataset according to the 2-Wasserstein distance. """
    if isinstance(dataset, FederatedDataset):
        partition_data = [dataset.load_partition(partition_id)["img"] for partition_id in range(num_partitions)]
        if isinstance(partition_data[0][0], str):
            partition_data = [[[int(x) for x in part.replace("[", "").replace("]", "").split(", ")] for part in partition] for partition in partition_data]
        labels = [dataset.load_partition(partition_id)["label"] for partition_id in range(num_partitions)]
    else:
        partition_data = [x for (x, y) in dataset]
        labels = [y for (x, y) in dataset]

    if needs_powertransform:
        pt = PowerTransformer(method='yeo-johnson', standardize=False)

        transformed_partitions = []
        for partition in partition_data:
            transformed_features = pt.fit_transform(partition)
            transformed_partitions.append(transformed_features)
    else:
        transformed_partitions = []
        for partition in partition_data:
            transformed_features = partition
            transformed_partitions.append(transformed_features)

    unique_labels = np.unique(labels[0])
    distances = []

    for class_label in unique_labels:
        for p1 in range(num_partitions):
            for p2 in range(p1 + 1, num_partitions):
                class_data_p1 = transformed_partitions[p1][labels[p1] == class_label]
                class_data_p2 = transformed_partitions[p2][labels[p2] == class_label]

                if class_data_p1.shape[0] == 0 or class_data_p2.shape[0] == 0:
                    continue

                matrix = ot.dist(class_data_p1, class_data_p2, metric='euclidean')

                # Uniform distribution weights for both distributions
                a = np.ones((class_data_p1.shape[0],)) / class_data_p1.shape[0]
                b = np.ones((class_data_p2.shape[0],)) / class_data_p2.shape[0]

                # Calculate the 2-Wasserstein distance
                wd = ot.emd2(a, b, matrix, numItermax=1000000)
                distances.append(wd)

    print(distances)

    # Compute the feature distribution skew as the average of these distances
    feature_distribution_skew = sum(distances) / len(distances) if len(distances) > 0 else 0

    return feature_distribution_skew


def one_hot_encode(labels):
    # Ensure labels are of long type
    labels = labels.long()

    # Create a tensor of zeros with shape (batch_size, num_classes)
    one_hot = torch.zeros(labels.size(0), settings.NUM_CLASSES, device=labels.device)

    # Scatter the labels into the one_hot tensor
    one_hot.scatter_(1, labels.unsqueeze(1), 1)

    return one_hot
