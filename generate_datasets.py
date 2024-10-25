import logging
import math
import random
import time
from itertools import cycle
from typing import List, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification

import settings
import library_adaptations
from utils import calculate_label_skew, calculate_feature_skew, calculate_label_skew_from_counts


def generate_label_skewed_counts(num_clients: int = 2,
                                 num_datapoints_per_client: int = settings.NUM_DATAPOINTS_PER_CLIENT,
                                 num_classes: int = 2,
                                 target_skewness: float = 0.5):
    counts_per_partition = []
    random.seed(settings.RANDOM_SEED)

    # Generate a label distribution of 0.0 by equally dividing the datapoints over the clients.
    for client in range(num_clients):
        client_ = {}
        for label in range(num_classes):
            client_[label] = num_datapoints_per_client // num_classes
        counts_per_partition.append(client_)

    # Set the previous labels, such that we do not keep swapping the same labels.
    previous = [0, 1]

    # Calculate the label skewness of the generated data and keep adjusting the data until the skewness is
    # within the target range.
    while True:
        label_skewness = calculate_label_skew_from_counts(counts_per_partition)

        # If the calculated skewness is smaller than the target skewness, we need to swap labels such that
        # both clients get more of each other's label.
        if label_skewness < (target_skewness - 0.025):
            client_1 = random.choice(range(num_clients))
            client_2 = random.choice([x for x in range(num_clients) if x != client_1])

            # Select the labels that are closest to the middle of the distribution to swap, such that both clients
            # get a more extreme (e.g. 0 or num_datapoints_per_client // num_classes) label.
            label_1 = np.argmin([abs(counts_per_partition[client_1][label] - (num_datapoints_per_client // num_classes)
                                     ) if label not in previous else np.inf for label in range(num_classes)])
            label_2 = np.argmin([abs(counts_per_partition[client_2][label] - (num_datapoints_per_client // num_classes)
                                     ) if label != label_1 else np.inf for label in range(num_classes)])

            # Swap at most 25 labels datapoints at once, or the minimum of the two labels that a client has.
            to_swap = min(counts_per_partition[client_1][label_1], 25, counts_per_partition[client_2][label_2])

            counts_per_partition[client_1][label_1] -= to_swap
            counts_per_partition[client_2][label_1] += to_swap

            counts_per_partition[client_1][label_2] += to_swap
            counts_per_partition[client_2][label_2] -= to_swap

            previous = [label_1, label_2]

        # Else, if the label skewness is higher than desired, we need to swap labels from the extremes of two clients
        # to get more towards the average each client should have.
        elif label_skewness > (target_skewness + 0.025):
            client_1 = random.choice(range(num_clients))
            client_2 = random.choice([x for x in range(num_clients) if x != client_1])

            # Now, select 2 labels based where the probability is equal to the label that is closest to
            # 0 or num_datapoints_per_client // num_classes: so the furthest away from the middle.
            label_1 = np.argmax([abs(counts_per_partition[client_1][label] - (num_datapoints_per_client // num_classes)
                                     ) if label not in previous else -np.inf for label in range(num_classes)])
            label_2 = np.argmax([abs(counts_per_partition[client_2][label] - (num_datapoints_per_client // num_classes)
                                     ) if label != label_1 else -np.inf for label in range(num_classes)])

            to_swap = min(counts_per_partition[client_2][label_1], 3, counts_per_partition[client_1][label_2])

            counts_per_partition[client_1][label_1] += to_swap
            counts_per_partition[client_2][label_1] -= to_swap

            counts_per_partition[client_1][label_2] -= to_swap
            counts_per_partition[client_2][label_2] += to_swap

        # If the label skewness falls within the margin of the target skewness, return the counts per partition and
        # actual skewness.
        else:
            return counts_per_partition, label_skewness


def generate_samples(counts_per_partition: List[Dict[int, int]], num_classes: int = settings.NUM_CLASSES):
    x, y = make_classification(n_samples=settings.NUM_DATAPOINTS_PER_CLIENT * len(counts_per_partition),
                               n_features=settings.NUM_FEATURES,
                               n_informative=settings.NUM_FEATURES - settings.NUM_FEATURES_REDUNDANT,
                               n_redundant=settings.NUM_FEATURES_REDUNDANT,
                               n_clusters_per_class=1,
                               n_classes=num_classes,
                               flip_y=0,
                               random_state=settings.RANDOM_SEED
                               )

    label_to_datapoints = dict()
    for x_, y_ in zip(x, y):
        if y_ not in label_to_datapoints:
            label_to_datapoints[y_] = []
        label_to_datapoints[y_].append(x_)

    for label in label_to_datapoints:
        random.shuffle(label_to_datapoints[label])

    partitions = []
    for client in counts_per_partition:
        client_x = []
        client_y = []
        for label in client:
            for i in range(client[label]):
                client_x.append(label_to_datapoints[label].pop())
                client_y.append(label)
        partitions.append((np.array(client_x), np.array(client_y)))

    return partitions


def generate_feature_skewed_dataset(num_clients: int = settings.NUM_CLIENTS,
                                    target_skewness: float = settings.TARGET_FEATURE_SKEWNESS):
    intra_sep = 1

    while True:
        x, y, z = tinker.make_classification(n_samples=settings.NUM_DATAPOINTS_PER_CLIENT * settings.NUM_CLIENTS,
                                             n_features=settings.NUM_FEATURES, n_informative=settings.NUM_FEATURES,
                                             n_redundant=0, n_clusters_per_class=num_clients, class_sep=1,
                                             intra_class_sep=intra_sep, n_classes=settings.NUM_CLASSES,
                                             random_state=settings.RANDOM_SEED)

        clients = []
        for client_idx in range(num_clients):
            mask = z == client_idx
            x_client = x[mask]
            y_client = y[mask]
            clients.append((x_client, y_client))

        skewness = calculate_feature_skew(clients, num_clients)
        print(f"Skewness: {skewness}, intra_sep: {intra_sep}")
        if skewness < target_skewness * 0.95:
            intra_sep *= 1.1
        elif skewness > target_skewness * 1.05:
            intra_sep *= 0.9
        else:
            break

    # plot_data(clients)
    print(f"Generated dataset with feature skewness {skewness}, "
          f"intra sep {intra_sep}")
    print(clients[0][0][:10], clients[0][1][:10])
    return clients, skewness


def generate_combined_skewed_dataset(num_clients: int = settings.NUM_CLIENTS,
                                     target_label_skewness: float = settings.TARGET_SKEWNESS,
                                     target_feature_skewness: float = settings.TARGET_FEATURE_SKEWNESS):
    counts_per_partition, calculated_label_skew = generate_label_skewed_counts(num_clients,
                                                                               settings.NUM_DATAPOINTS_PER_CLIENT,
                                                                               settings.NUM_CLASSES,
                                                                               target_label_skewness)

    intra_sep = 1

    while True:
        x, y, z = tinker.make_classification(n_samples=round((settings.NUM_DATAPOINTS_PER_CLIENT * num_clients) * 1.05),
                                             # Generate 5% more data to compensate for label difference
                                             n_features=settings.NUM_FEATURES, n_informative=settings.NUM_FEATURES,
                                             n_redundant=0, n_clusters_per_class=num_clients, class_sep=30,
                                             intra_class_sep=intra_sep, n_classes=settings.NUM_CLASSES,
                                             random_state=settings.RANDOM_SEED,
                                             weights=list(np.ones(settings.NUM_CLASSES) / settings.NUM_CLASSES))

        label_to_datapoints = dict()
        for x_, y_ in zip(x, y):
            if y_ not in label_to_datapoints:
                label_to_datapoints[y_] = []
            label_to_datapoints[y_].append(x_)

        for label in label_to_datapoints:
            random.shuffle(label_to_datapoints[label])

        partitions = []
        for client in counts_per_partition:
            client_x = []
            client_y = []
            for label in client:
                for i in range(client[label]):
                    client_x.append(label_to_datapoints[label].pop())
                    client_y.append(label)
            partitions.append((np.array(client_x), np.array(client_y)))

        calculated_feature_skew = calculate_feature_skew(partitions, num_clients)
        print(f"Skewness: {calculated_feature_skew}, intra_sep: {intra_sep}")
        if calculated_feature_skew < target_feature_skewness * 0.95:
            intra_sep *= 1.1
        elif calculated_feature_skew > target_feature_skewness * 1.05:
            intra_sep *= 0.9
        else:
            break
    return partitions, calculated_label_skew, calculated_feature_skew
