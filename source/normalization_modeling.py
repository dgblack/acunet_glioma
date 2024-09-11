# General
from pathlib import Path
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
import pickle
import json

# File IO
import pickle
import scipy.io as sio

# Pytorch
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from .pytorch_utils import ResidualBlock

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

class PpIXPhantomDataset(Dataset):
    """Face Landmarks dataset."""
    _VALID_MODES = {'uv', 'white', 'uvwhite', 'uvwhite_stk'}
    def __init__(self, phantomData_matfile, labels_pklfile, mode, concentrations=None, transform=None):
        """
        Args:
            phantomData_matfile (string or pathlike): Path to the phantom data.
            labels_pklfile (string or pathlike): Path to labels file. Must have
                the same number of entries as index or concentrations 
            mode (string): {'uv', 'white', 'uvwhite', 'uvwhite_stk'}
            concentration (list): concentrations to use
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data = sio.loadmat(phantomData_matfile)
        self.concentrations = data['concentration'].flatten()
        self.index = data['index'].astype(int) - 1 # to shift to python indexing
        self.uvSpectra = data['uvSpectra']
        self.wSpectra = data['wSpectra']

        self.labels = pickle.load(open(labels_pklfile, 'rb'))
        if len(self.labels) == len(self.index):
            self.labels = self.labels.reshape((-1, 1))
        elif len(self.labels) == len(self.concentrations):
            labels = [self.labels[self.concentrations[i[0]]] for i in self.index]
            longest = max(len(l) for l in labels if l is not None)
            labels = [np.repeat(np.nan, longest) if l is None else l for l in labels]
            self.labels = np.array(labels).astype(float)
        else:
              raise ValueError('labels file must have same number of entries as index or concentrations ')

        if mode not in self._VALID_MODES: 
              raise ValueError('dataset mode must be in {}'.format(self._VALID_MODES))

        # drop nan rows and cast
        non_nan = ~np.isnan(self.labels).any(axis=1)
        self.index = self.index[non_nan].astype(float)
        self.uvSpectra = self.uvSpectra[non_nan].astype(float)
        self.wSpectra = self.wSpectra[non_nan].astype(float)
        self.labels = self.labels[non_nan].astype(float)

        # TODO in out shape
        self.mode = mode
        self.transform = transform
       

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.mode == 'uv':
            spectrum = self.uvSpectra[idx]
        elif self.mode == 'white':
            spectrum = self.wSpectra[idx]
        elif self.mode == 'uvwhite':
            spectrum = np.concatenate((self.uvSpectra[idx], self.wSpectra[idx]))
        elif self.mode == 'uvwhite_stk':
            spectrum = np.stack((self.uvSpectra[idx], self.wSpectra[idx]))
        else:
            raise AttributeError('Dataset declared with invalid mode')

        # sample = {'spectrum': spectrum, 'label': self.labels[idx]}
        sample = [spectrum, self.labels[idx]]

        if self.transform:
            sample = self.transform(sample)

        return sample

class DenseNet(nn.Module):
    '''
    Dense MLP Model for using conventional suggestions for dense nets 
    for hidden layer sizes with only linear and relu layers.

    Default behavior if parameters unspecified is one hidden layer
    with size = mean(input_size, output_size)

    "n sum, for most problems, one could probably get decent performance 
    (even without a second optimization step) by setting the hidden layer 
    configuration using just two rules: 
      (i) the number of hidden layers equals one; and 
      (ii) the number of neurons in that layer is the mean of the neurons 
    in the input and output layers."

    Eg. 
      [wData | uvData] --> k scaling factor (620 -> 1)
      [wData] --> k (310 -> 1)
      []

    '''
    def __init__(self, 
        inputSz=310, 
        outputSz=1, 
        hidden_layers=None, 
        dropout: float=0.0
        ):
        super(DenseNet, self).__init__()
        if not hidden_layers:
            hidden_layers = [int(np.mean([inputSz, outputSz]))]
        
        # Define all layer sizes
        layer_sizes = [inputSz] + hidden_layers + [outputSz]
        layers = []

        # Create Layers
        for i in range(len(layer_sizes)-2):
            layers.append(nn.BatchNorm1d(layer_sizes[i]))
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.BatchNorm1d(layer_sizes[-2]))
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        # Create sequence stack
        self.dense_stack = nn.Sequential(*layers)

        # metadata
        self.input_size = inputSz
        self.output_size = outputSz
        self.hidden_layers = hidden_layers
        self.dropout = dropout

    def forward(self, x):
        return self.dense_stack(x)

    def save_arch(self, path):
        with(open(path, 'wb+')) as f:
            metadata = {
                'inputSz': self.input_size,
                'outputSz': self.output_size,
                'hidden_layers': self.hidden_layers,
                'dropout': self.dropout
            }
            pickle.dump(metadata, f)

    @staticmethod
    def load_arch(path):
        with(open(path, 'rb')) as f:
            arch = pickle.load(f)
        return DenseNet(**arch)

class RegressionConv1D(nn.Module):
    def __init__(self):
        super(RegressionConv1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(224, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.reshape(-1, 1, 620)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class SpectralUnet(nn.Module):
    def __init__(self):
        super(SpectralUnet, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.transconv1 = nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1)
        self.transconv2 = nn.ConvTranspose1d(32, 1, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(155, 310)
        # self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.reshape(-1, 1, 620)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x  = self.transconv1(x)
        x = self.transconv2(x)
        x = self.fc1(x)
        x = x.reshape(-1, 310)
        # print(x.shape)
        return x

class ResConv1D(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResConv1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(9920, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, output_size)
        self.input_size = input_size

    def forward(self, x):
        x = x.reshape(-1, 1, self.input_size)
        x1 = self.conv1(x)
        x = self.conv2(x1) + x1
        x = self.pool1(x)
        x2 = self.conv3(x)
        x = self.conv4(x2) + x2
        x = self.pool2(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))

    def save_config(self, filepath):
        config = {
            "input_size": self.input_size,
            "output_size": self.fc2.out_features
        }
        with open(filepath, "w+") as f:
            json.dump(config, f)

class StackedResConv1D(nn.Module):
    def __init__(self, input_size, input_channels, output_size):
        super(StackedResConv1D, self).__init__()
        self.res1 = ResidualBlock(input_channels, input_size, layers=3, num_filters=32, kernel_size=3, stride=1, padding=1)
        self.res2 = ResidualBlock(*self.res1.output_shape, layers=3, num_filters=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.res2.output_shape = (self.res2.output_shape[0], self.res2.output_shape[1]//2)
        self.res3 = ResidualBlock(*self.res2.output_shape, layers=3, num_filters=64, kernel_size=5, stride=1, padding=2)
        self.res4 = ResidualBlock(*self.res3.output_shape, layers=3, num_filters=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        self.res4.output_shape = (self.res4.output_shape[0], self.res4.output_shape[1]//2)
        self.fc1 = nn.Linear(np.prod(self.res4.output_shape), 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.input_channels = input_channels
        self.input_size = input_size

    def forward(self, x):
        # x = x.reshape(-1, self.input_channels, self.input_size)
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool1(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.pool2(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return x

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))

    def save_config(self, filepath):
        config = {
            "input_size": self.input_size,
            "output_size": self.fc2.out_features
        }
        with open(filepath, "w+") as f:
            json.dump(config, f)

# if __name__ == '__main__':
#     model = StackedResConv1D(310, 2, 5)
#     X = model(torch.rand(1, 2, 310))
#     print(X.shape)