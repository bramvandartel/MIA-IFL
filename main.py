from comet_ml import Experiment, OfflineExperiment

import logging
import math
import os
import re
import pickle
from typing import List, Tuple

import torch.autograd
from flwr.common import Metrics
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset

import data_loader
import model
import settings
from clients import FlowerClient, AttackerFlowerClient
import flwr as fl

if settings.OFFLINE:
    experiment = OfflineExperiment(
        api_key=settings.COMETML_API_KEY,
        project_name="Final project",
        workspace=settings.COMET_WORKSPACE,
    )
else:
    experiment = Experiment(
        api_key=settings.COMETML_API_KEY,
        project_name="Final project",
        workspace=settings.COMET_WORKSPACE,
    )
experiment.set_name(settings.JOB_ID)
tmp = None

# Function for the weighted average in FedAvg
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# The client function is used to initialise the FlowerClient and the attacker.
def client_fn(cid: str, to_client=True):
    if to_client:
        net = class_().to(settings.DEVICE)
    else:
        net = class_().to(settings.DEVICE2)
    train_loader = train[int(cid)]
    val_loader = val[int(cid)]

    if cid == '0':
        client = AttackerFlowerClient(net=net, train_loader=train_loader, val_loader=val_loader,
                                      criterion=settings.LOSS_FUNCTION)
    else:
        client = FlowerClient(net, train_loader, val_loader, criterion=settings.LOSS_FUNCTION)
    if to_client:
        return client.to_client()
    return client


hyper_params = {
    "num_clients": settings.NUM_CLIENTS,
    "num_rounds": settings.NUM_ROUNDS,
    "target_model": settings.TARGET_MODEL,
    "target_skewness": None if settings.SKEWED == "feature" else settings.TARGET_SKEWNESS,
    "num_datapoints_per_client": settings.NUM_DATAPOINTS_PER_CLIENT,
    "num_features": settings.NUM_FEATURES,
    "num_features_redundant": settings.NUM_FEATURES_REDUNDANT,
    "dataset": settings.DATASET if not settings.GENERATE else "Generated",
    "attack_model_num_rounds_input": settings.ATTACK_MODEL_NUM_ROUNDS_INPUT,
    "optimizer": settings.OPTIMIZER,
    "batch_size": settings.BATCH_SIZE,
    "loss_function": settings.LOSS_FUNCTION,
    "attack_model_epochs": settings.ATTACK_MODEL_EPOCHS,
    "skewed": settings.SKEWED,
    "attack_model_batch_size": settings.ATTACK_MODEL_BATCH_SIZE,
    "num_classes": settings.NUM_CLASSES,
    "train_partition": settings.PERCENTAGE_OF_TRAIN_DATA_TO_USE,
    "test_partition": settings.PERCENTAGE_OF_TEST_DATA_TO_USE,
    "test_set_from": "Target",
    "target_feature_skewness": None if settings.SKEWED == "label" else settings.TARGET_FEATURE_SKEWNESS,
}

experiment.log_parameters(hyper_params)

torch.autograd.set_detect_anomaly(True)
summary = []

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

directory = os.path.join(settings.STORAGE_PATH)

print(f"Running Target Model Training on {settings.DEVICE} and Attack Model Training on {settings.DEVICE2}")

with open(f"{settings.STORAGE_PATH}/info.txt", 'w+') as target:
    with open("settings.py", "r") as source:
        target.writelines(source.readlines())

accuracy = 0
while accuracy < 0.5:
    if settings.GENERATE:
        train, val, skewness = data_loader.load_generated_data(target_skewness=settings.TARGET_SKEWNESS,
                                                           num_clients=settings.NUM_CLIENTS,
                                                           num_datapoints_per_client=settings.NUM_DATAPOINTS_PER_CLIENT,
                                                           num_classes=settings.NUM_CLASSES)
        summary.append(f"Generated data with skewness {skewness}")
        experiment.log_parameter(f"{settings.SKEWED.lower()}-skewness", skewness)
    else:
        if settings.DATASET.lower() == "cifar10":
            train, val = data_loader.load_cifar10_datasets()
        elif settings.DATASET.lower() == "cifar100":
            train, val = data_loader.load_cifar100_datasets()
        elif (settings.DATASET.lower() in ['texas100', 'purchase100', 'adult', 'heart'] or 'splitted' in
              settings.DATASET):
            train, val, summary = data_loader.load_custom_datasets(settings.DATASET.lower(), summary, experiment)
        else:
            raise ValueError("Invalid dataset ", settings.DATASET)

    class_ = getattr(model, settings.TARGET_MODEL)

    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=settings.NUM_CLIENTS,
        min_evaluate_clients=settings.NUM_CLIENTS,
        min_available_clients=settings.NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
    )

    # Specify the resources of the clients.
    client_resources = {"num_cpus": max(1, math.floor(settings.NUM_CPU // settings.NUM_CLIENTS)),
                        "num_gpus": settings.NUM_GPU / settings.NUM_CLIENTS}

    filename = f'trained_target_model-{settings.TARGET_MODEL}-{"Generate" if settings.GENERATE else settings.DATASET}-{settings.NUM_ROUNDS}-{settings.NUM_CLIENTS}.pickle'

    # Check if the file exists
    if os.path.isfile(filename):
        # If the file exists, delete it
        os.remove(filename)

    # Start simulation
    tmp = fl.simulation.start_simulation(
        client_fn=client_fn,
        clients_ids=range(settings.NUM_CLIENTS),
        num_clients=settings.NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=settings.NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={'ignore_reinit_error': True, 'num_cpus': settings.NUM_CPU, 'num_gpus': settings.NUM_GPU,
                       '_temp_dir': settings.RAY_PATH}
    )

    pattern = r'\(100, (\d+\.\d+)\)'
    match = re.search(pattern, str(tmp))

    if not match:
        raise NotImplementedError("Whut...")

    else:
        accuracy = float(match.group(1))

    if not settings.GENERATE:
        accuracy = accuracy if accuracy > 0.5 else 1

if tmp is not None:
    summary.append(tmp)
experiment.log_parameter("Target Model", summary)

# After simulation, open the weights and initialise the attacker again.
with open(f'trained_target_model-{settings.TARGET_MODEL}-{"Generate" if settings.GENERATE else settings.DATASET}-{settings.NUM_ROUNDS}-{settings.NUM_CLIENTS}.pickle', 'rb') as f:
    weights = pickle.load(f)


attacker = client_fn("0", to_client=False)
attacker.experiment = experiment
target_client_id = 1
target_client = client_fn(str(target_client_id), to_client=False)


if not settings.DIFFERENT_NON_MEMBERS:  # If we don't want a third partition, then do this:
    # Retrieve data from the attacker.
    trained_dataset = train[0].dataset  # Attacker train data, e.g. train members
    target_trained_dataset = train[target_client_id].dataset  # Target train data, e.g. test members
    validation_dataset = val[0].dataset  # Attacker validation data, e.g. train non-members
    target_validation_dataset = val[1].dataset  # Target validation data, e.g. test non-members

    # To get an equal number, we want to have the partition with the least datapoints.
    cnt = min(len(validation_dataset), len(target_validation_dataset), len(trained_dataset), len(target_trained_dataset))

    # First, we want to get the non-members into test and train.

    # From the attacker validation data, e.g. train non-members, retrieve the TRAIN_PARTION * cnt.
    training_non_members, _ = random_split(validation_dataset,
                                           [round(settings.PERCENTAGE_OF_TRAIN_DATA_TO_USE * cnt), len(validation_dataset) - round(settings.PERCENTAGE_OF_TRAIN_DATA_TO_USE * cnt)])

    # From the target validation data, e.g. test non-members, retrieve the TEST_PARTION * cnt.
    test_non_members, _ = random_split(target_validation_dataset,
                                       [round(settings.PERCENTAGE_OF_TEST_DATA_TO_USE * cnt), len(target_validation_dataset) - round(settings.PERCENTAGE_OF_TEST_DATA_TO_USE * cnt)])

    # Then, for the members, we want to get them into test and train as well.

    # From the attacker training data, e.g. train members, retrieve TRAIN_PARTION * cnt.
    training_members, _ = random_split(trained_dataset, [round(settings.PERCENTAGE_OF_TRAIN_DATA_TO_USE * cnt),
                                                         len(trained_dataset) - round(settings.PERCENTAGE_OF_TRAIN_DATA_TO_USE * cnt)])

    # From the target training data, e.g. test members, retrieve TEST_PARTION * cnt.
    test_members, _ = random_split(target_trained_dataset,
                                   [round(settings.PERCENTAGE_OF_TEST_DATA_TO_USE * cnt),
                                    len(target_trained_dataset) - round(settings.PERCENTAGE_OF_TEST_DATA_TO_USE * cnt)])
else:  # Now, we do want a third partition!
    if settings.PERC == 1:  # If we want all test non-members to be from party C.
        non_members = val[settings.NUM_CLIENTS].dataset  # Non-members from Party C, thus test non-members.
        # train_non_members = val[0].dataset  # Train non-members from attacker's validation dataset.
        test_members = train[1].dataset  # Test members from target's training dataset.
        train_members = train[0].dataset  # Train members from attacker's training dataset.

        train_non_members, test_non_members = random_split(non_members, [round(len(non_members) * 0.5), round(len(non_members) * 0.5)])

        cnt = min(len(test_non_members), len(train_non_members), len(train_members), len(test_members))

        # First, we want to get the non-members into test and train.

        # From the attacker validation data, e.g. train non-members, retrieve the TRAIN_PARTION * cnt.
        training_non_members, _ = random_split(train_non_members,
                                               [round(settings.PERCENTAGE_OF_TRAIN_DATA_TO_USE * cnt),
                                                len(train_non_members) - round(settings.PERCENTAGE_OF_TRAIN_DATA_TO_USE * cnt)])

        # From the target validation data, e.g. test non-members, retrieve the TEST_PARTION * cnt.
        test_non_members, _ = random_split(test_non_members,
                                           [round(settings.PERCENTAGE_OF_TEST_DATA_TO_USE * cnt),
                                            len(test_non_members) - round(settings.PERCENTAGE_OF_TEST_DATA_TO_USE * cnt)])

        # Then, for the members, we want to get them into test and train as well.

        # From the attacker training data, e.g. train members, retrieve TRAIN_PARTION * cnt.
        training_members, _ = random_split(train_members, [round(settings.PERCENTAGE_OF_TRAIN_DATA_TO_USE * cnt),
                                                           len(train_members) - round(
                                                               settings.PERCENTAGE_OF_TRAIN_DATA_TO_USE * cnt)])

        # From the target training data, e.g. test members, retrieve TEST_PARTION * cnt.
        test_members, _ = random_split(test_members,
                                       [round(settings.PERCENTAGE_OF_TEST_DATA_TO_USE * cnt),
                                        len(test_members) - round(settings.PERCENTAGE_OF_TEST_DATA_TO_USE * cnt)])

    elif settings.PERC == 0.5:  # In the case that we want the third partition to be 50/50 split of known data:
        part_1 = val[settings.NUM_CLIENTS].dataset  # First, we pick the validation dataset of party C. This is for test non-members.
        part_2 = val[1].dataset  # Then, we pick the validation dataset of the Target. This is for test non-members.

        # Find the minimum length between part_1 and part_2
        min_length = min(len(part_1), len(part_2))

        # Create subsets from both datasets using the smallest length
        part_1_subset = Subset(part_1, list(range(min_length)))
        part_2_subset = Subset(part_2, list(range(min_length)))

        # Combine the two subsets with equal sizes
        non_members = ConcatDataset([part_1_subset, part_2_subset])
        train_non_members, test_non_members = random_split(non_members, [round(len(non_members) * 0.5), round(len(non_members) * 0.5)])
        # We now have a non-members dataset containing 50% of seen distribution and 50% of unseen distribution.

        # train_non_members = val[0].dataset  # Train non-members is the members from the attacker's validation dataset.
        train_members = train[0].dataset  # Train members is the members from the attacker's training dataset.
        test_members = train[1].dataset  # Train members is the members from the target's training dataset.


        # Now, we want to split!
        cnt = min(len(test_non_members), len(train_non_members), len(train_members),
                  len(test_members))

        # First, we want to get the non-members into test and train.

        # From the attacker validation data, e.g. train non-members, retrieve the TRAIN_PARTION * cnt.
        training_non_members, _ = random_split(train_non_members,
                                               [round(settings.PERCENTAGE_OF_TRAIN_DATA_TO_USE * cnt),
                                                len(train_non_members) - round(settings.PERCENTAGE_OF_TRAIN_DATA_TO_USE * cnt)])

        # From the target validation data, e.g. test non-members, retrieve the TEST_PARTION * cnt.
        test_non_members, _ = random_split(test_non_members,
                                           [round(settings.PERCENTAGE_OF_TEST_DATA_TO_USE * cnt),
                                            len(test_non_members) - round(settings.PERCENTAGE_OF_TEST_DATA_TO_USE * cnt)])

        # Then, for the members, we want to get them into test and train as well.

        # From the attacker training data, e.g. train members, retrieve TRAIN_PARTION * cnt.
        training_members, _ = random_split(train_members, [round(settings.PERCENTAGE_OF_TRAIN_DATA_TO_USE * cnt),
                                                           len(train_members) - round(
                                                               settings.PERCENTAGE_OF_TRAIN_DATA_TO_USE * cnt)])

        # From the target training data, e.g. test members, retrieve TEST_PARTION * cnt.
        test_members, _ = random_split(test_members,
                                       [round(settings.PERCENTAGE_OF_TEST_DATA_TO_USE * cnt),
                                        len(test_members) - round(settings.PERCENTAGE_OF_TEST_DATA_TO_USE * cnt)])

    else:
        part_1 = val[
            settings.NUM_CLIENTS].dataset  # First, we pick the validation dataset of party C. This is for test non-members.
        part_2 = val[1].dataset  # Then, we pick the validation dataset of the Target. This is for test non-members.

        # Find the minimum length between part_1 and part_2
        min_length = min(len(part_1), len(part_2))

        # Create subsets from both datasets using the smallest length
        part_1_subset = Subset(part_1, list(range(round(min_length * settings.PERC))))
        part_2_subset = Subset(part_2, list(range(round(min_length * (1 - settings.PERC)))))

        # Combine the two subsets
        non_members = ConcatDataset([part_1_subset, part_2_subset])
        train_non_members, test_non_members = random_split(non_members, [math.ceil(len(non_members) * 0.5),
                                                                         math.floor(len(non_members) * 0.5)])

        # train_non_members = val[0].dataset  # Train non-members is the members from the attacker's validation dataset.
        train_members = train[0].dataset  # Train members is the members from the attacker's training dataset.
        test_members = train[1].dataset  # Train members is the members from the target's training dataset.

        # Now, we want to split!
        cnt = min(len(test_non_members), len(train_non_members), len(train_members),
                  len(test_members))

        # First, we want to get the non-members into test and train.

        # From the attacker validation data, e.g. train non-members, retrieve the TRAIN_PARTION * cnt.
        training_non_members, _ = random_split(train_non_members,
                                               [round(settings.PERCENTAGE_OF_TRAIN_DATA_TO_USE * cnt),
                                                len(train_non_members) - round(settings.PERCENTAGE_OF_TRAIN_DATA_TO_USE * cnt)])

        # From the target validation data, e.g. test non-members, retrieve the TEST_PARTION * cnt.
        test_non_members, _ = random_split(test_non_members,
                                           [round(settings.PERCENTAGE_OF_TEST_DATA_TO_USE * cnt),
                                            len(test_non_members) - round(settings.PERCENTAGE_OF_TEST_DATA_TO_USE * cnt)])

        # Then, for the members, we want to get them into test and train as well.

        # From the attacker training data, e.g. train members, retrieve TRAIN_PARTION * cnt.
        training_members, _ = random_split(train_members, [round(settings.PERCENTAGE_OF_TRAIN_DATA_TO_USE * cnt),
                                                           len(train_members) - round(
                                                               settings.PERCENTAGE_OF_TRAIN_DATA_TO_USE * cnt)])

        # From the target training data, e.g. test members, retrieve TEST_PARTION * cnt.
        test_members, _ = random_split(test_members,
                                       [round(settings.PERCENTAGE_OF_TEST_DATA_TO_USE * cnt),
                                        len(test_members) - round(settings.PERCENTAGE_OF_TEST_DATA_TO_USE * cnt)])

training_members = DataLoader(training_members, batch_size=1, shuffle=True)
test_members = DataLoader(test_members, batch_size=1, shuffle=True)
training_non_members = DataLoader(training_non_members, batch_size=1, shuffle=True)
test_non_members = DataLoader(test_non_members, batch_size=1, shuffle=True)

del train, val


# Let the attacker perform the actual attack.
summ, model = attacker.attack(weights, training_members, training_non_members, test_members, test_non_members,
                              experiment)
summary.extend(summ)

experiment.log_parameter("Summary", summary)
