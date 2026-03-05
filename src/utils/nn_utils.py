import torch
import torch.nn as nn
import numpy as np

def weights_init_(layer, init="kaiming", activation="relu"):
    if isinstance(layer, torch.nn.Linear):
        gain = torch.nn.init.calculate_gain(activation)

        if init == "xavier_uniform":
            torch.nn.init.xavier_uniform_(layer.weight, gain=gain)
        elif init == "xavier_normal":
            torch.nn.init.xavier_normal_(layer.weight, gain=gain)
        elif init == "uniform":
            torch.nn.init.uniform_(layer.weight) / layer.in_features
        elif init == "normal":
            torch.nn.init.normal_(layer.weight) / layer.in_features
        elif init == "orthogonal":
            torch.nn.init.orthogonal_(layer.weight)
        elif init == "zeros":
            torch.nn.init.zeros_(layer.weight)
        elif init == "kaiming_uniform" or init == "default" or init is None:
            # PyTorch default
            return
        else:
            raise NotImplementedError(f"init {init} not implemented yet")

        if isinstance(layer, torch.nn.Linear) and layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)

def init_layers(layers, init_scheme):
    def fill_weights(layers, init_fn):
        for i in range(len(layers)):
            init_fn(layers[i].weight)

    if init_scheme.lower() == "xavier_uniform":
        fill_weights(layers, nn.init.xavier_uniform_)
    elif init_scheme.lower() == "xavier_normal":
        fill_weights(layers, nn.init.xavier_normal_)
    elif init_scheme.lower() == "uniform":
        fill_weights(layers, nn.init.uniform_)
    elif init_scheme.lower() == "normal":
        fill_weights(layers, nn.init.normal_)
    elif init_scheme.lower() == "orthogonal":
        fill_weights(layers, nn.init.orthogonal_)
    elif init_scheme is None:
        # Use PyTorch default
        return

def _get_activation(activation):
    if activation.lower() == "relu":
        act = nn.ReLU()
    elif activation.lower() == "tanh":
        act = nn.Tanh()
    else:
        raise ValueError(f"unknown activation {activation}")

    return act


def hard_update(target, source):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            # Use in-place operations mul_ and add_ to avoid copying tensor data
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * param.data)