from typing import List

import torch
from torch import nn


class NN_DQN(nn.Module):
    """MLP for DQN market maker

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation_function,
        list_weights: List[int],
    ):
        """Construct MLP

        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            activation_function (_type_): activation function
            list_weights (List[int]): weights of the hidden layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.list_weights = list_weights

        layers = []
        layers.append(nn.Linear(input_dim, list_weights[0]))
        layers.append(nn.BatchNorm1d(list_weights[0]))
        layers.append(activation_function)

        # Add additional hidden layers
        num_layers = len(self.list_weights)
        for i in range(num_layers - 1):
            layers.append(nn.Linear(list_weights[i], list_weights[i + 1]))
            layers.append(nn.BatchNorm1d(list_weights[i + 1]))
            layers.append(activation_function)

        layers.append(nn.Linear(list_weights[num_layers - 1], output_dim))

        self.linear_relu_stack = nn.Sequential(*layers)
        self.double()

    def forward(self, x):
        """Forward  pass

        Parameters
        ----------
        x : _type_
            input tensor

        Returns
        -------
        _type_
            _description
        """
        output = self.linear_relu_stack(x)
        return output
