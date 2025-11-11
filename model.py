import torch.nn as nn
import pandas as pd


class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, reg_config, activation="relu"):
        super(FeedForwardNN, self).__init__()
        self.reg_config = reg_config

        activation_funcs = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
            "elu": nn.ELU()
        }

        if activation not in activation_funcs:
            raise ValueError(f"Invalid activation function: {activation}. Choose from {list(activation_funcs.keys())}")

        self.activation_layer = activation_funcs[activation]

        # Explicitly define input layer
        self.input_layer = nn.Linear(input_size, hidden_layers[0])
        self.input_layer.bias.data.zero_()                # Set bias to 0
        self.input_layer.bias.requires_grad = False       # Prevent training it

        # Define rest of the network
        layers = []
        prev_size = hidden_layers[0]
        for hidden in hidden_layers[1:]:
            layers.append(nn.Linear(prev_size, hidden))
            layers.append(self.activation_layer)
            if reg_config["dropout"] > 0:
                layers.append(nn.Dropout(reg_config["dropout"]))
            prev_size = hidden

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x):
        x = self.activation_layer(self.input_layer(x))
        if self.reg_config["dropout"] > 0:
            x = nn.functional.dropout(x, p=self.reg_config["dropout"], training=self.training)
        x = self.hidden_layers(x)
        return self.output_layer(x)