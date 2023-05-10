# model.py
# Contains our NeuralNet class
import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        """
        Apply a linear combination of weights and biases
        """
        self.l1 = nn.Linear(input_size, hidden_size)  # Input layer
        self.l2 = nn.Linear(hidden_size, hidden_size) # Hidden layer
        self.l3 = nn.Linear(hidden_size, num_classes) # Output layer
        self.relu = nn.ReLU() # Rectified linear activation function applied to layers

    def forward(self, x):
        """
        Do a forward pass
        """
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # No activation and no softmax on last output
        return out