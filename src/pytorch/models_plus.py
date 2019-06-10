import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)


class Cnn_5layers_AvgPooling(nn.Module):

    def __init__(self, classes_num, activation):
        super(Cnn_5layers_AvgPooling, self).__init__()

        self.activation = activation

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.fc)

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))

        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))

        x = F.relu_(self.bn3(self.conv3(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))

        x = F.relu_(self.bn4(self.conv4(x)))
        x = F.avg_pool2d(x, kernel_size=(1, 1))
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        x = torch.mean(x, dim=3)  # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)  # (batch_size, feature_maps)
        x = self.fc(x)

        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)

        elif self.activation == 'sigmoid':
            output = F.sigmoid(x)

        return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn_9layers_AvgPooling(nn.Module):

    def __init__(self, classes_num, activation):
        super(Cnn_9layers_AvgPooling, self).__init__()

        self.activation = activation
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(1)
        self.bn3 = nn.BatchNorm2d(1)
        self.bn4 = nn.BatchNorm2d(1)
        self.bn5 = nn.BatchNorm2d(1)
        self.bn6 = nn.BatchNorm2d(1)
        self.conv_block1 = ConvBlock(in_channels=6, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)
        init_bn(self.bn5)
        init_bn(self.bn6)

    def forward(self, data, data_left, data_right, data_side, data_harmonic, data_percussive):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x1 = self.bn1(data[:, None, :, :])
        x2 = self.bn2(data_left[:, None, :, :])
        x3 = self.bn3(data_right[:, None, :, :])
        x4 = self.bn4(data_side[:, None, :, :])
        x5 = self.bn5(data_harmonic[:, None, :, :])
        x6 = self.bn6(data_percussive[:, None, :, :])
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        '''(batch_size, 3, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        x = torch.mean(x, dim=3)  # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)  # (batch_size, feature_maps)
        x = self.fc(x)

        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)

        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)

        return output

class Cnn_9layers_AvgPooling_mix(nn.Module):

    def __init__(self, classes_num, activation):
        super(Cnn_9layers_AvgPooling_mix, self).__init__()

        self.activation = activation
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(1)
        self.bn3 = nn.BatchNorm2d(1)
        self.bn4 = nn.BatchNorm2d(1)
        self.bn5 = nn.BatchNorm2d(1)
        self.bn6 = nn.BatchNorm2d(1)
        self.conv_block1_1 = ConvBlock(in_channels=2, out_channels=64)
        self.conv_block1_2 = ConvBlock(in_channels=2, out_channels=64)
        self.conv_block1_3 = ConvBlock(in_channels=2, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=192, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, classes_num, bias=True)
        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)
        init_bn(self.bn5)
        init_bn(self.bn6)

    def forward(self, data, data_left, data_right, data_side, data_harmonic, data_percussive):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        a1 = self.bn1(data[:, None, :, :])
        a2 = self.bn2(data_left[:, None, :, :])
        a3 = self.bn3(data_right[:, None, :, :])
        a4 = self.bn4(data_side[:, None, :, :])
        a5 = self.bn5(data_harmonic[:, None, :, :])
        a6 = self.bn6(data_percussive[:, None, :, :])
        x1 = torch.cat((a1, a4), dim=1)
        x2 = torch.cat((a2, a3), dim=1)
        x3 = torch.cat((a5, a6), dim=1)
        '''(batch_size, 3, times_steps, freq_bins)'''

        x1 = self.conv_block1_1(x1, pool_size=(2, 2), pool_type='avg')
        x2 = self.conv_block1_2(x2, pool_size=(2, 2), pool_type='avg')
        x3 = self.conv_block1_3(x3, pool_size=(2, 2), pool_type='avg')
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        x = torch.mean(x, dim=3)  # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)  # (batch_size, feature_maps)
        x = self.fc(x)

        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)

        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)

        return output


class Cnn_9layers_MaxPooling(nn.Module):
    def __init__(self, classes_num, activation):

        super(Cnn_9layers_MaxPooling, self).__init__()

        self.activation = activation

        self.conv_block1 = ConvBlock(in_channels=6, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, data, data_left, data_right, data_side, data_harmonic, data_percussive):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x1 = data[:, None, :, :]
        x2 = data_left[:, None, :, :]
        x3 = data_right[:, None, :, :]
        x4 = data_side[:, None, :, :]
        x5 = data_harmonic[:, None, :, :]
        x6 = data_percussive[:, None, :, :]
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        '''(batch_size, 3, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='max')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        x = torch.mean(x, dim=3)  # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)  # (batch_size, feature_maps)
        x = self.fc(x)

        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)

        elif self.activation == 'sigmoid':
            output = F.sigmoid(x)

        return output
