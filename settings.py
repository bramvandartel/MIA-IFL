import os

import torch

# One time settings

COMET_WORKSPACE = "xxx"  # CometML workspace
HUGGINGFACE_USERNAME = "xxx"  # Huggingface username

# Frequently used
GENERATE = False
SKEWED = "label"  # Choices ["label", "feature", "both"]
NUM_CLIENTS = 2
NUM_ROUNDS = 100  # For training the target model
TARGET_MODEL = "HeartNet"  # Choose from model.py
DATASET = "heart_va_ch_splitted_extra_partition"  # From huggingface datasets in HUGGINGFACE_USERNAME space.

# When GENERATE=True
TARGET_SKEWNESS = 0  # Target label skewness when GENERATE=True and SKEWED="label" or SKEWED="both"
TARGET_FEATURE_SKEWNESS = 0  # Target feature skewness when GENERATE=True and SKEWED="feature" or SKEWED="both"
NUM_DATAPOINTS_PER_CLIENT = 500  # When GENERATE=True
NUM_FEATURES = 75  # When GENERATE=True
NUM_FEATURES_REDUNDANT = 0  # When GENERATE=True
RANDOM_SEED = 43  # When GENERATE=True


PERCENTAGE_OF_TRAIN_DATA_TO_USE = 1  # Between 0 and 1
PERCENTAGE_OF_TEST_DATA_TO_USE = 1  # Between 0 and 1

NUM_CPU = int(os.environ.get('SLURM_CPUS_PER_TASK')) if str(os.environ.get('SLURM_CPUS_PER_TASK')).isnumeric() else 8
NUM_GPU = 0

ATTACK_MODEL_NUM_ROUNDS_INPUT = 1  # Number of rounds from target model to input. Can potentially also be a list of rounds, e.g. [25, 50, 75, 100].
OPTIMIZER = torch.optim.Adam  # Optimizer

# Other settings
ALPHA = 0.1  # Only used for CIFAR datasets
BATCH_SIZE = 64  # Batch size of Target Model
LOSS_FUNCTION = torch.nn.CrossEntropyLoss()  # Loss function
ATTACK_MODEL_EPOCHS = 30  # Number of epochs for the Attack Model
ATTACK_MODEL_BATCH_SIZE = 4  # Batch size of the Attack Model
NUM_CLASSES = 2  # Number of classes in the dataset (or to generate)

COMETML_API_KEY = "xxx"  # CometML API key

# Least used settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # To train Target Model on
DEVICE2 = torch.device("cpu")  # To train Attack Model on
JOB_ID = os.environ.get('SLURM_JOBID')
STORAGE_PATH = f'output/{JOB_ID}'

OFFLINE = False  # Does not use Comet if Offline.
DIFFERENT_NON_MEMBERS = False  # Interesting for Humphries attack.
PERC = 1    # Choices [0.25, 0.50, 0.75, 1]

RAY_PATH = f""  # Location of RAY.

if DATASET == "heart_va_ch_splitted_extra_partition":
    HEARTNET_FEATURES = 35
elif "heart_ch_hg" in DATASET:
    HEARTNET_FEATURES = 29
elif "heart_cl_va" in DATASET:
    HEARTNET_FEATURES = 38
elif "heart_cl_hg" in DATASET:
    HEARTNET_FEATURES = 29
elif "heart_va_ch" in DATASET:
    HEARTNET_FEATURES = 32
else:
    HEARTNET_FEATURES = 44
