import pickle
from typing import List, Dict, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import Scalar
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from tqdm import tqdm

import attack_model
import settings
from data_loader import CombinedDataLoader
from model import FLNet
from settings import DEVICE
from utils import one_hot_encode


class FlowerClient(fl.client.NumPyClient):
    """
    Default implementation of a client participating in the Federated Learning protocol.
    """
    net: FLNet
    train_loader: DataLoader
    val_loader: DataLoader
    criterion: _Loss

    def __init__(self, net: FLNet, train_loader: DataLoader, val_loader: DataLoader,
                 criterion: _Loss):
        """
        Initialise a standard FL client.
        :param net: Network to train
        :param train_loader: Dataset of the client
        :param val_loader: Validation dataset of the client
        :param criterion: Loss function
        """
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion

    def get_parameters(self, config) -> List[np.ndarray]:
        """
        Return the parameters. Needed by Flwr.
        :param config:
        :return:
        """
        return self.net.get_parameters()

    def set_parameters(self, parameters: List[np.ndarray]):
        """
        Set the parameters. Needed by Flwr.
        :param parameters:
        :return:
        """
        self.net.set_parameters(parameters)

    def fit(self, parameters: List[np.ndarray], config):
        """
        Train the model given the parameters from the server and return the updated parameters and number of samples.
        :param parameters: Model parameters
        :param config:
        :return:
        """
        self.net.set_parameters(parameters)
        self.train()
        return self.net.get_parameters(), len(self.train_loader), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model given the test loader.
        :param parameters:
        :param config:
        :return:
        """
        self.net.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), len(self.val_loader), {"accuracy": float(accuracy)}

    def train(self, optimizer=settings.OPTIMIZER, epochs=1, verbose=False):
        """
        Train the model given an optimizer.
        :param optimizer:
        :param epochs:
        :param verbose:
        :return:
        """
        self.net.train()
        optimizer = optimizer(self.net.parameters(), lr=1e-3)
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for batch in self.train_loader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                optimizer.zero_grad()
                output, _ = self.net(images)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()

                # Metrics
                epoch_loss += loss
                total += labels.size(0)
                correct += (torch.max(output.data, 1)[1] == labels).sum().item()
            epoch_loss /= len(self.train_loader.dataset)
            epoch_acc = correct / total
            if verbose:
                print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    def test(self):
        """
        Test the model given a test dataset.
        :return:
        """
        correct, total, loss = 0, 0, 0.0
        self.net.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                output, _ = self.net(images)
                loss += self.criterion(output, labels).item()
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        loss /= len(self.val_loader)
        accuracy = correct / total
        return loss, accuracy


class AttackerFlowerClient(FlowerClient):
    """
    Extension to the FlowerClient which has attacker functionalities. An instance can simulate an attack in the FL
    protocol.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_model = None
        self.experiment = None

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Override the evaluate function, as to collect data during the execution of the protocol.
        :param parameters:
        :param config:
        :return:
        """

        # Load the stored pickle file, if exists. Otherwise, create a new dictionary. Clients are stateless in Flwr.
        try:
            data = pickle.load(open(f'trained_target_model-{settings.TARGET_MODEL}-{"Generate" if settings.GENERATE else settings.DATASET}-{settings.NUM_ROUNDS}-{settings.NUM_CLIENTS}.pickle', 'rb'))
        except FileNotFoundError:
            print("No model of previous round found. Creating new one.")
            data = {'round': 1, 'model': {}}

        data['model'][data['round']] = parameters
        data['round'] += 1
        pickle.dump(data, open(f'trained_target_model-{settings.TARGET_MODEL}-{"Generate" if settings.GENERATE else settings.DATASET}-{settings.NUM_ROUNDS}-{settings.NUM_CLIENTS}.pickle', 'wb+'))

        return super().evaluate(parameters, config)

    def gather_results(self, data_points, label, weights):
        for _, datapoint in data_points:
            data = self.gather_features(datapoint, weights)
            data['label'] = label
            yield data

    def attack(self, weights: Dict, training_members, training_non_members, test_members, test_non_members, experiment):
        summary = []
        self.net.to(settings.DEVICE2)

        # Shuffle members and non-members.
        training_loader = CombinedDataLoader({1: training_members, 0: training_non_members}, weights, self)
        testing_loader = CombinedDataLoader({1: test_members, 0: test_non_members}, weights, self)

        summary.extend(self.train_attack_model(training_loader, testing_loader, experiment))
        with open(f"{settings.STORAGE_PATH}/trained_attack_model.pickle", "wb") as f:
            pickle.dump(self.attack_model, f)
        print("Done training, going to start testing...")
        acc = self.test_attack_model(testing_loader, experiment, settings.ATTACK_MODEL_EPOCHS)
        summary.append(f"Final accuracy: {acc:.4f} after {settings.ATTACK_MODEL_EPOCHS} epochs.")
        return summary, self.attack_model


    def gather_features(self, datapoint, weights):
        # Simulate all rounds in the protocol
        true_label = one_hot_encode(datapoint['label']).to(settings.DEVICE2).requires_grad_(True)
        attack_features = {'true_label': true_label.requires_grad_(True), 'loss': [], 'layers': {}, 'gradients': {}}
        weights = weights['model']

        # Once this supported selection of rounds, e.g. [25, 50, 75, 100] would pick these rounds.
        # No clue if that still works.
        if isinstance(settings.ATTACK_MODEL_NUM_ROUNDS_INPUT, int):
            rnds = list(weights.keys())[-1 * settings.ATTACK_MODEL_NUM_ROUNDS_INPUT:]
        else:
            rnds = settings.ATTACK_MODEL_NUM_ROUNDS_INPUT


        for rnd in rnds:
            # 1. Initialise the model to the round
            self.net.set_parameters(weights[rnd])
            self.net.train()
            optimizer = settings.OPTIMIZER(self.net.parameters())
            # 2. Calculate all hidden layer values for given datapoint
            optimizer.zero_grad()
            prediction, layers = self.net(datapoint['img'].to(settings.DEVICE2).requires_grad_(True))

            # 3. Calculate the gradients of all weights w.r.t. datapoint

            # 3.1 Calculate the loss and gradients of the given datapoint
            loss = self.criterion(prediction.requires_grad_(True), datapoint['label'].type(torch.LongTensor).to(settings.DEVICE2))
            loss.backward()
            optimizer.step()
            attack_features['loss'].append(loss.item())

            for layer_id in range(len(layers)):
                if layer_id not in attack_features['layers'].keys():
                    attack_features['layers'][layer_id] = []
                attack_features['layers'][layer_id].append(layers[layer_id].detach())

            # 3.2 Filter out all bias layers
            weight_layers = []
            for name, _ in self.net.named_parameters():
                if 'weight' in name:
                    weight_layers.append(name.split('.')[0])

            # 3.3 Append all weight-gradient layers.
            for gradient_id in range(len(weight_layers)):
                if gradient_id not in attack_features['gradients'].keys():
                    attack_features['gradients'][gradient_id] = []
                attack_features['gradients'][gradient_id].append(
                    getattr(self.net, weight_layers[gradient_id]).weight.grad.detach())
        attack_features['loss'] = torch.tensor(attack_features['loss'])
        return attack_features

    def train_attack_model(self, train_loaders, test_loaders, experiment):
        summary = []

        # Do some first time calculations
        for idx, item in enumerate(train_loaders):
            first_datapoint = {'x_output': item['layers'], 'x_true_label': item.get('true_label', None).clone().detach().requires_grad_(True).to(settings.DEVICE2), 'x_loss': item['loss'], 'x_grad': item['gradients'],
                           'x_in_training_data': item['label']}
            break
        
        cuda_count = 1
        self.attack_model = attack_model.AttackModel(first_datapoint, num_cuda=cuda_count) if cuda_count > 0 else attack_model.AttackModel(first_datapoint)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(params=self.attack_model.parameters(), lr=0.0001)

        for epoch in range(settings.ATTACK_MODEL_EPOCHS):
            print("Epoch ", epoch)
            for batch_idx, item in tqdm(enumerate(train_loaders), total=len(train_loaders)):
                x_hidden_layers = item['layers']
                x_true_label = item.get('true_label', None).clone().detach().requires_grad_(True).to(settings.DEVICE2)
                x_loss = item['loss']
                x_gradients = item['gradients']
                y_label = item['label']
                if y_label.size(0) != settings.ATTACK_MODEL_BATCH_SIZE:
                    continue

                optimizer.zero_grad()
                outputs = self.attack_model(x_hidden_layers, x_loss, x_true_label, x_gradients)
                loss = criterion(torch.squeeze(outputs), y_label.squeeze(-1).type(torch.FloatTensor).to(settings.DEVICE2))

                # Backward and optimize
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch + 1}/{settings.ATTACK_MODEL_EPOCHS}], Loss: {loss.item():.4f}')
            acc = self.test_attack_model(test_loaders, experiment, epoch + 1)
            summary.append(f"Epoch [{epoch + 1}/{settings.ATTACK_MODEL_EPOCHS}], Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")
            self.experiment.log_metric("Accuracy", acc, epoch=epoch+1)
            self.experiment.log_metric("Loss", loss.item(), epoch=epoch+1)
        return summary

    def test_attack_model(self, test_loaders, experiment, i):
        # Evaluation
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        for batch_idx, item in tqdm(enumerate(test_loaders), total=len(test_loaders)):
            x_hidden_layers = item['layers']
            x_true_label = item.get('true_label', None).clone().detach().requires_grad_(True).to(settings.DEVICE2)
            x_loss = item['loss']
            x_gradients = item['gradients']
            target = item['label']
            if target.size(0) != settings.ATTACK_MODEL_BATCH_SIZE:
                continue
            with torch.no_grad():
                self.attack_model.eval()
                outputs = self.attack_model(x_hidden_layers, x_loss, x_true_label, x_gradients)
                predicted = (outputs >= 0.5).float().squeeze()
                target = target.float().squeeze()

                y_pred.extend(list(int(x) for x in predicted.cpu().numpy()))
                y_true.extend(list(int(x) for x in target.cpu().numpy()))

                total += target.size(0)
                correct += (predicted.cpu() == target.cpu()).sum().item()
                print(f"Correct: {(predicted.cpu() == target.cpu()).sum().item()}, Total: {target.size(0)}, predictions: {predicted}, truth: {target}")

        print(f'Accuracy: {100 * correct / total}%')
        experiment.log_confusion_matrix(y_true=y_true, y_predicted=y_pred, labels=["Non-member", "Member"], title=f"Confusion matrix epoch {i}", epoch=i)
        return 100 * correct / total