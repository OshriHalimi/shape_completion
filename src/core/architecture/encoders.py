import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------------------------------
#                                               Encoders
# ----------------------------------------------------------------------------------------------------------------------
class ShapeEncoder(nn.Module):
    def __init__(self, code_size=1024, in_channels=3, dense=True):
        super().__init__()
        self.code_size = code_size
        self.in_channels = in_channels

        if dense:
            features = DensePointNetFeatures(self.code_size, self.in_channels)
        else:
            features = PointNetFeatures(self.code_size, self.in_channels)

        self.encoder = nn.Sequential(
            features,
            nn.Linear(self.code_size, self.code_size),
            nn.BatchNorm1d(self.code_size),
            nn.ReLU()
        )

    def init_weights(self):
        bn_gamma_mu = 1.0
        bn_gamma_sigma = 0.02
        bn_betta_bias = 0.0

        nn.init.normal_(self.encoder[2].weight, mean=bn_gamma_mu, std=bn_gamma_sigma)  # weight=gamma
        nn.init.constant_(self.encoder[2].bias, bn_betta_bias)  # bias=betta

    # Input: Batch of Point Clouds : [b x num_vertices X in_channels]
    # Output: The global feature vector : [b x code_size]
    def forward(self, shape):
        return self.encoder(shape)


# ----------------------------------------------------------------------------------------------------------------------
#                                               Encoders
# ----------------------------------------------------------------------------------------------------------------------

class PointNetFeatures(nn.Module):
    def __init__(self, code_size, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, code_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(code_size)

    def init_weights(self):
        conv_mu = 0.0
        conv_sigma = 0.02
        bn_gamma_mu = 1.0
        bn_gamma_sigma = 0.02
        bn_betta_bias = 0.0

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=conv_mu, std=conv_sigma)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, mean=bn_gamma_mu, std=bn_gamma_sigma)  # weight=gamma, bias=betta
                nn.init.constant_(m.bias, bn_betta_bias)

    # noinspection PyTypeChecker
    def forward(self, x):
        # Input: Batch of Point Clouds : [b x num_vertices X in_channels]
        # Output: The global feature vector : [b x code_size]
        x = x.transpose(2, 1).contiguous()  # [b x in_channels x num_vertices]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))  # [B x 128 x n]
        x = self.bn3(self.conv3(x))  # [B x code_size x n]
        x, _ = torch.max(x, 2)  # [B x code_size]
        return x


class DensePointNetFeatures(nn.Module):
    def __init__(self, code_size, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(in_channels + 64, 128, 1)
        self.conv3 = nn.Conv1d(in_channels + 64 + 128, code_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(code_size)

    def init_weights(self):
        conv_mu = 0.0
        conv_sigma = 0.02
        bn_gamma_mu = 1.0
        bn_gamma_sigma = 0.02
        bn_betta_bias = 0.0

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=conv_mu, std=conv_sigma)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, mean=bn_gamma_mu, std=bn_gamma_sigma)  # weight=gamma, bias=betta
                nn.init.constant_(m.bias, bn_betta_bias)

    # noinspection PyTypeChecker
    def forward(self, x):
        # Input: Batch of Point Clouds : [b x num_vertices X in_channels]
        # Output: The global feature vector : [b x code_size]
        x = x.transpose(2, 1).contiguous()  # [b x in_channels x num_vertices]
        y = F.relu(self.bn1(self.conv1(x)))  # [B x 64 x n]
        z = F.relu(self.bn2(self.conv2(torch.cat((x, y), 1))))  # [B x 128 x n]
        z = self.bn3(self.conv3(torch.cat((x, y, z), 1)))  # [B x code_size x n]
        z, _ = torch.max(z, 2)  # [B x code_size]
        return z
