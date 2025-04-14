# Name: Ting Li; Duo Xu. 
# Date: March 27.
# Travel day used: 2 days totally, 1 day for Ting and 1 day for Duo. 
# This file contains the definition of the experiment convolutional neural network and fixed gabor filter of extension. 

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class ExperimentNetwork(nn.Module):
    def __init__(self, num_conv_filters=10, num_fc_nodes=64, dropout_rate=0.5, use_gabor=False):
        super(ExperimentNetwork, self).__init__()
        self.use_gabor = use_gabor

        if self.use_gabor:
            self.conv1 = self._create_gabor_layer(num_filters=num_conv_filters)
            for param in self.conv1.parameters():
                param.requires_grad = False  # holding the first layer constant.
        else:
            self.conv1 = nn.Conv2d(1, num_conv_filters, kernel_size=5)

        self.conv2 = nn.Conv2d(num_conv_filters, 20, kernel_size=5)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.fc1 = nn.Linear(320, num_fc_nodes)
        self.fc2 = nn.Linear(num_fc_nodes, 10)

    def _create_gabor_layer(self, num_filters):
        gabor_kernels = []
        for i in range(num_filters):
            theta = i * np.pi / num_filters
            kernel = cv2.getGaborKernel((5, 5), 2.0, theta, 3.0, 0.5, 0, ktype=cv2.CV_32F)
            kernel = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)
            gabor_kernels.append(kernel)

        weights = torch.cat(gabor_kernels, dim=0)
        conv = nn.Conv2d(1, num_filters, kernel_size=5, bias=False)
        conv.weight = nn.Parameter(weights, requires_grad=False)
        return conv

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
