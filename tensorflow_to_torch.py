import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorflow.keras.models import load_model

# Load the Keras .h5 model
keras_model = load_model("model.h5")
print("Loaded Keras model from model.h5")

# Define the equivalent PyTorch model
class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=5)  # no stride
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 1 * 18, 100)  # Manually computed size after conv layers
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.flatten(x)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return x

# Instantiate the PyTorch model
pytorch_model = PyTorchModel()

# Function to transfer weights
def transfer_conv_weights(keras_layer, torch_layer):
    weights = keras_layer.get_weights()
    torch_layer.weight.data = torch.tensor(weights[0]).permute(3, 2, 0, 1).float()
    torch_layer.bias.data = torch.tensor(weights[1]).float()

def transfer_dense_weights(keras_layer, torch_layer):
    weights = keras_layer.get_weights()
    torch_layer.weight.data = torch.tensor(weights[0].T).float()
    torch_layer.bias.data = torch.tensor(weights[1]).float()

# Map each layer
transfer_conv_weights(keras_model.layers[0], pytorch_model.conv1)
transfer_conv_weights(keras_model.layers[1], pytorch_model.conv2)
transfer_conv_weights(keras_model.layers[2], pytorch_model.conv3)
transfer_conv_weights(keras_model.layers[3], pytorch_model.conv4)

transfer_dense_weights(keras_model.layers[5], pytorch_model.fc1)
transfer_dense_weights(keras_model.layers[6], pytorch_model.fc2)
transfer_dense_weights(keras_model.layers[7], pytorch_model.fc3)
transfer_dense_weights(keras_model.layers[8], pytorch_model.fc4)

# Save the PyTorch model
torch.save(pytorch_model.state_dict(), "model.pth")
print("Converted and saved as model.pth")
