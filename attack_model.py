from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import settings


# Initialise weights according to settings of Nasr.
def initialise_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class FcnComponent(nn.Module):

    def __init__(self, input_size: int = 128, layer_size: int = 128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fcn = nn.Sequential(
            nn.Linear(input_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        ).to(settings.DEVICE2)
        self.fcn.apply(initialise_weights)

    def forward(self, x):
        x = self.fcn(x)
        return x


class CnnComponent(nn.Module):

    def __init__(self, output_layers: int, kernel_size: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1000, kernel_size=(1, kernel_size), stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        ).to(settings.DEVICE2)
        self.fcn = FcnComponent(input_size=output_layers, layer_size=1024).to(settings.DEVICE2)

    def forward(self, x):
        x = x.view(1, x.shape[0]*x.shape[1], -1)
        x = self.cnn(x.to(settings.DEVICE2))
        x = x.view(x.size()[0], -1)
        return self.fcn(x.to(settings.DEVICE2))


class AttackModel(nn.Module):

    def __init__(self, first_datapoint, num_cuda=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_cuda = num_cuda
        encoder_component_input = 0

        x_output_shapes = []
        for stacked_x in first_datapoint['x_output'].values():
            shape = stacked_x.shape
            transformed_x = stacked_x.view(1, (shape[0] // num_cuda) * shape[1] * shape[2], -1)  # New shape of format (NUM_ROUNDS * BATCH_SIZE * 1, REST)
            x_output_shapes.append(transformed_x.shape[2])
        encoder_component_input += len(first_datapoint['x_output'].keys()) * 64
        self.hidden_layer_components = nn.ModuleList([FcnComponent(input_size=x, layer_size=128).to(settings.DEVICE2) for x in x_output_shapes])

        # Once this supported selection of rounds, e.g. [25, 50, 75, 100] would pick these rounds.
        # No clue if that still works.
        if isinstance(settings.ATTACK_MODEL_NUM_ROUNDS_INPUT, int):
            self.loss_component = FcnComponent(settings.ATTACK_MODEL_NUM_ROUNDS_INPUT).to(settings.DEVICE2)
        else:
            self.loss_component = FcnComponent(len(settings.ATTACK_MODEL_NUM_ROUNDS_INPUT)).to(settings.DEVICE2)
        encoder_component_input += 64
        self.true_label_component = FcnComponent(first_datapoint['x_true_label'].shape[-1]).to(settings.DEVICE2)
        encoder_component_input += 64

        modules = []
        for stacked_x in first_datapoint['x_grad'].values():
            shape = stacked_x.shape
            transformed_x = stacked_x.view(1, shape[0] * shape[1], shape[2], -1)
            outp = transformed_x.shape[1]*transformed_x.shape[2]*(transformed_x.shape[3]-9) if num_cuda < 2 else (transformed_x.shape[1]*transformed_x.shape[2]*(transformed_x.shape[3]-9))
            if outp < 0:
                outp = outp * -1
            modules.append(CnnComponent(outp).to(settings.DEVICE2))

        self.gradient_components = nn.ModuleList(modules)

        encoder_component_input += 4000 * len(first_datapoint['x_grad'].keys())

        # Once this supported selection of rounds, e.g. [25, 50, 75, 100] would pick these rounds.
        # No clue if that still works.
        if isinstance(settings.ATTACK_MODEL_NUM_ROUNDS_INPUT, int):
            encoder_component_input += 64 * len(first_datapoint['x_grad'].keys()) * (settings.ATTACK_MODEL_NUM_ROUNDS_INPUT - 1)
        else:
            encoder_component_input += 64 * len(first_datapoint['x_grad'].keys()) * (
                        len(settings.ATTACK_MODEL_NUM_ROUNDS_INPUT) - 1)

        # This part once worked automatically, but now it needs to be set manually.
        if settings.TARGET_MODEL == "AlexNet":
            encoder_component_input = 256640
        else:
            if settings.ATTACK_MODEL_BATCH_SIZE == 10:
                if settings.GENERATE:
                    encoder_component_input = 32448
                elif settings.DATASET == 'heart_splitted' or settings.DATASET == 'students_splitted':
                    encoder_component_input = 32448
                else:
                    encoder_component_input = 33728

            else:
                if settings.ATTACK_MODEL_BATCH_SIZE == 10:
                    encoder_component_input = 33728
                elif settings.ATTACK_MODEL_BATCH_SIZE == 4:
                    encoder_component_input = 80448
                elif settings.ATTACK_MODEL_BATCH_SIZE == 16:
                    encoder_component_input = 20448
                else:
                    encoder_component_input = 160448
        self.encoder = EncoderComponent(encoder_component_input).to(settings.DEVICE2)
        self.apply(initialise_weights)

    def forward(self, x_hidden_layers, x_loss, x_true_label, x_gradients):

        # For every hidden layer, call the forward function of the Fcn with the rounds stacked.
        result_x_hidden_layers = []
        for i in range(len(x_hidden_layers)):
            stacked_x = x_hidden_layers[i]
            shape = stacked_x.shape
            transformed_x = stacked_x.view(shape[0] * shape[1] * shape[2], -1)

            # Once this supported selection of rounds, e.g. [25, 50, 75, 100] would pick these rounds.
            # No clue if that still works.
            if isinstance(settings.ATTACK_MODEL_NUM_ROUNDS_INPUT, int):
                result_x_hidden_layers.append(self.hidden_layer_components[i](transformed_x.to(settings.DEVICE2)).view(settings.ATTACK_MODEL_BATCH_SIZE, settings.ATTACK_MODEL_NUM_ROUNDS_INPUT, 1, -1).to(settings.DEVICE2))
            else:
                result_x_hidden_layers.append(self.hidden_layer_components[i](transformed_x.to(settings.DEVICE2)).view(settings.ATTACK_MODEL_BATCH_SIZE, len(settings.ATTACK_MODEL_NUM_ROUNDS_INPUT), 1, -1).to(settings.DEVICE2))

        result_x_hidden_layers = torch.cat(result_x_hidden_layers, dim=1)
        result_x_hidden_layers.requires_grad_(True)

        result_x_loss = self.loss_component(x_loss.to(settings.DEVICE2)).to(settings.DEVICE2)
        result_x_loss.requires_grad_(True)

        result_x_true_label = self.true_label_component(x_true_label.to(settings.DEVICE2)).requires_grad_(True).squeeze().to(settings.DEVICE2)
        result_x_true_label.requires_grad_(True)

        result_x_gradients = []
        for layer_id in range(len(x_gradients)):
            stacked_x = x_gradients[layer_id]
            shape = stacked_x.shape
            transformed_x = stacked_x.view(shape[0] * shape[1], shape[2], -1)

            # Once this supported selection of rounds, e.g. [25, 50, 75, 100] would pick these rounds.
            # No clue if that still works.
            if isinstance(settings.ATTACK_MODEL_NUM_ROUNDS_INPUT, int):
                res = self.gradient_components[layer_id](transformed_x.to(settings.DEVICE2)).view(settings.ATTACK_MODEL_BATCH_SIZE, settings.ATTACK_MODEL_NUM_ROUNDS_INPUT, 1, -1).to(settings.DEVICE2)
            else:
                res = self.gradient_components[layer_id](transformed_x.to(settings.DEVICE2)).view(settings.ATTACK_MODEL_BATCH_SIZE, len(settings.ATTACK_MODEL_NUM_ROUNDS_INPUT), 1, -1).to(settings.DEVICE2)
            result_x_gradients.append(res.requires_grad_(True))

        result_x_gradients = torch.cat(result_x_gradients, dim=1)
        result_x_gradients.requires_grad_(True)

        final_result = torch.cat((result_x_hidden_layers.view(settings.ATTACK_MODEL_BATCH_SIZE, -1), result_x_loss.view(settings.ATTACK_MODEL_BATCH_SIZE, -1), result_x_true_label.view(settings.ATTACK_MODEL_BATCH_SIZE, -1), result_x_gradients.view(settings.ATTACK_MODEL_BATCH_SIZE, -1)), dim=1)
        final_result.requires_grad_(True)
        encoded = self.encoder(final_result.to(settings.DEVICE2))
        return encoded


class EncoderComponent(nn.Module):

    def __init__(self, in_features: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fcn = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fcn(x)
