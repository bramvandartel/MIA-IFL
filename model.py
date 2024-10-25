from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import settings


class FLNet(nn.Module):
    """
    Default implementation of a Neural Network that is supported by the Flwr framework.
    """

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Sets the parameters of the model according to the input given.
        :param parameters: List of np arrays containing weights.
        :return: None
        """
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)

    def get_parameters(self) -> List[np.ndarray]:
        """
        Returns the parameters of the Neural Network.
        :return: List of np arrays containing the parameters
        """
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def __str__(self):
        return self.__class__.__name__


class Net(FLNet):

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, settings.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        outputs = []
        x = self.conv1(x)
        outputs.append(x)
        x = F.relu(x, inplace=False)
        x = self.pool(x)
        x = self.conv2(x)
        outputs.append(x)
        x = F.relu(x, inplace=False)
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        outputs.append(x)
        x = F.relu(x, inplace=False)
        x = self.fc2(x)
        outputs.append(x)
        x = F.relu(x, inplace=False)
        x = self.fc3(x)
        outputs.append(x)
        x = F.relu(x, inplace=False)
        return x, outputs


class Texas100Net(FLNet):

    def __init__(self):
        super(Texas100Net, self).__init__()
        self.fc1 = nn.Linear(6169, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 100)

    def forward(self, x):
        outputs = []
        x = torch.relu(self.fc1(x))
        outputs.append(x)
        x = torch.relu(self.fc2(x))
        outputs.append(x)
        x = torch.relu(self.fc3(x))
        outputs.append(x)
        x = torch.relu(self.fc4(x))
        outputs.append(x)
        x = torch.relu(self.fc5(x))
        outputs.append(x)
        return x, outputs


class Purchase100Net(FLNet):

    def __init__(self):
        super(Purchase100Net, self).__init__()
        self.fc1 = nn.Linear(600, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 100)

    def forward(self, x):
        outputs = []
        x = torch.relu(self.fc1(x))
        outputs.append(x)
        x = torch.relu(self.fc2(x))
        outputs.append(x)
        x = torch.relu(self.fc3(x))
        outputs.append(x)
        x = torch.relu(self.fc4(x))
        outputs.append(x)
        x = torch.relu(self.fc5(x))
        outputs.append(x)
        return x, outputs


class HeartNet(FLNet):

    def __init__(self):
        super(HeartNet, self).__init__()
        self.fc1 = nn.Linear(settings.HEARTNET_FEATURES, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 2)

    def forward(self, x):
        outputs = []
        x = torch.relu(self.fc1(x))
        outputs.append(x)
        x = torch.relu(self.fc2(x))
        outputs.append(x)
        x = torch.relu(self.fc3(x))
        outputs.append(x)
        x = torch.relu(self.fc4(x))
        outputs.append(x)
        x = torch.relu(self.fc5(x))
        outputs.append(x)
        return x, outputs


class GeneratedDatasetNetwork(FLNet):
    def __init__(self):
        super(GeneratedDatasetNetwork, self).__init__()
        self.fc1 = nn.Linear(settings.NUM_FEATURES, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, settings.NUM_CLASSES)

    def forward(self, x):
        outputs = []
        x = torch.relu(self.fc1(x))
        outputs.append(x)
        x = torch.relu(self.fc2(x))
        outputs.append(x)
        x = torch.relu(self.fc3(x))
        outputs.append(x)
        x = torch.relu(self.fc4(x))
        outputs.append(x)
        x = torch.relu(self.fc5(x))
        outputs.append(x)
        return x, outputs


class StudentNet(FLNet):

    def __init__(self):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(50, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 2)

    def forward(self, x):
        outputs = []
        x = torch.relu(self.fc1(x))
        outputs.append(x)
        x = torch.relu(self.fc2(x))
        outputs.append(x)
        x = torch.relu(self.fc3(x))
        outputs.append(x)
        x = torch.relu(self.fc4(x))
        outputs.append(x)
        x = torch.relu(self.fc5(x))
        outputs.append(x)
        return x, outputs