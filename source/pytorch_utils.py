from torch.utils.data import Dataset
from torch import nn
import torch

class InceptionBlock(nn.Module):
    def __init__(self, input_channels, num_filters=16):
        super(InceptionBlock, self).__init__()
        self.conv1x1 = nn.Conv1d(input_channels, num_filters, kernel_size=1)
        self.conv3x3_1 = nn.Conv1d(input_channels, num_filters, kernel_size=1)
        self.conv3x3_2 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv5x5_1 = nn.Conv1d(input_channels, num_filters, kernel_size=1)
        self.conv5x5_2 = nn.Conv1d(num_filters, num_filters, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(input_channels, num_filters, kernel_size=1)
        self.bn = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.conv1x1(x)
        out2 = self.conv3x3_1(x)
        out2 = self.conv3x3_2(out2)
        out3 = self.conv5x5_1(x)
        out3 = self.conv5x5_2(out3)
        out4 = self.pool(x)
        out4 = self.conv_pool(out4)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, input_channels, input_size=310, layers=2, num_filters=16, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.convlayers = nn.ModuleList()
        
        # First layer:
        conv = nn.Conv1d(input_channels, num_filters, kernel_size=kernel_size, stride=stride, padding=padding)
        bn = nn.BatchNorm1d(num_filters)
        relu = nn.ReLU()
        unit = nn.ModuleDict({
            'conv': conv,
            'bn': bn,
            'relu': relu
        })
        self.convlayers.append(unit)
        output_size = ((input_size - kernel_size + 2 * padding) // stride) + 1
        self.first_layer = input_channels != num_filters
        
        #subsequent layers
        for _ in range(1, layers):
            conv = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, stride=stride, padding=padding)
            bn = nn.BatchNorm1d(num_filters)
            relu = nn.ReLU()
            unit = nn.ModuleDict({
                'conv': conv,
                'bn': bn,
                'relu': relu
            })
            self.convlayers.append(unit)
            output_size = ((output_size - kernel_size + 2 * padding) // stride) + 1

        self.output_shape = (num_filters, output_size)

    def forward(self, x):
        residual = x
        resnet_start = 0
        if self.first_layer:
            unit = self.convlayers[0]
            x = unit['conv'](x)
            x = unit['bn'](x)
            x = unit['relu'](x)
            residual = x
            resnet_start = 1
        for unit in self.convlayers[resnet_start:-1]:
            x = unit['conv'](x)
            x = unit['bn'](x)
            x = unit['relu'](x)
        unit = self.convlayers[-1]
        x = unit['conv'](x)
        x = unit['bn'](x)
        x += residual
        out = unit['relu'](x)
        return out
    
class TensorDataset(Dataset):
    def __init__(self, X, y):
        if X.shape[0] != y.shape[0]: raise ValueError('X, and Y must have the same number of samples')
        self.X = X
        self.y = y
        
    def __len__(self):
        return self.X.size()[0]

    def __getitem__(self, idx):
        return self.X[idx,...], self.y[idx,...]