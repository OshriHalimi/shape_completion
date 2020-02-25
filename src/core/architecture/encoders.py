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
#                                           Encoder - DGCNN
# ----------------------------------------------------------------------------------------------------------------------
class ShapeEncoderDGCNN(nn.Module):
    def __init__(self, in_channels, code_size, k, device):
        super().__init__()
        self.k = k
        self.in_channels = in_channels
        self.code_size = code_size
        self.dev = device

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm1d(self.code_size)

        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256, self.code_size, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(self.code_size * 2, self.code_size, bias=False)
        self.bn6 = nn.BatchNorm1d(self.code_size)

    def get_graph_feature(self, x, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = knn(x, k=self.k)  # (batch_size, num_points, k)
        idx_base = torch.arange(0, batch_size, device=self.dev).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        _, num_dims, _ = x.size()
        x = x.transpose(2,
                        1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, self.k, 1)
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
        return feature

    def init_weights(self):
        pass

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        batch_size = x.size(0)
        x = self.get_graph_feature(x)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = self.get_graph_feature(x1)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = self.get_graph_feature(x2)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = self.get_graph_feature(x3)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)  # [B x 2 * code_size]

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # [B x code_size]
        return x


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


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
