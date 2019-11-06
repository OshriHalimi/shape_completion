from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

# UTILITIES
# OH: V
class STN3d(nn.Module):  # OH: An alignment network T-Net (Probably)
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):  # OH: input is a batch of point clouds, output is a batch of 3x3 rotation matrix
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x % (2 * np.pi)
        x = torch.squeeze(x)
        r = torch.eye(3).expand(x.shape[0],3,3).clone()
        r[:, 0, 0] = torch.cos(x)
        r[:, 0, 2] = -torch.sin(x)
        r[:, 2, 0] = torch.sin(x)
        r[:, 2, 2] = torch.cos(x)
        r = r.contiguous().cuda()
        return r



# OH: V
class PointNetfeat(nn.Module):
    # OH: The network takes a (batch) point cloud as an input
    # (dimensions: [B x 3 x num_points_input] )
    # and outputs a (batch) global feature vector
    # (dimensions: [B x bottleneck_size] )
    # or point features (dimensions: [B x (64 + bottleneck_size) x num_points_input] )
    def __init__(self, global_feat=True, trans=False, bottleneck_size = 1024, num_input_channels = 3):
        # OH: If trans is True alignment transformation is applied to the point cloud; If global_feat is True,
        # a single global feature vector is returned for each point cloud; If global_feat is False, the global
        # feature vector is copied for each point and concatenated with the point features; In this case,
        # the final structure is similar to the initial input of the segmentation network in PointNet (See diagram
        # https://arxiv.org/pdf/1612.00593.pdf)
        super(PointNetfeat, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.num_input_channels = num_input_channels
        self.trans = trans
        self.global_feat = global_feat

        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(self.num_input_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.bottleneck_size, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size)


    def forward(self, x):
        #if self.trans:
        #    trans = self.stn(x)  # OH: batch transformation; [B x 3 x 3].
        #    # OH: In practice the point cloud is rotated by the transpose of trans
        #    x = x.transpose(2, 1)  # OH: [B x 3 x n] --> [B x n x 3]
        #    x = torch.bmm(x, trans)  # OH: batch matrix-matrix product
        #    x = x.transpose(2, 1).contiguous()  # OH: [B x n x 3] --> [B x 3 x n]

        num_points = x.shape[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x  # OH: [B x 64 x n]
        x = F.relu(self.bn2(self.conv2(x)))  # OH: [B x 128 x n]
        x = self.bn3(self.conv3(x))  # OH: [B x bottleneck_size x n]
        x, _ = torch.max(x, 2)  # OH: [B x bottleneck_size]
        x = x.view(-1, self.bottleneck_size)  # OH: redundant line ???
        if self.global_feat:
            return x
        else:
            x = x.view(-1, self.bottleneck_size, 1).repeat(1, 1, num_points)
            return torch.cat([x, pointfeat], 1)


# OH: V
class PointGenCon(nn.Module):
    # OH: Input is a (batch) structure concatenating the point cloud coordinates and a global code (repeated for each point)
    # (input dimensions: [B x (3 + global_code_size) x num_points] )
    # The output is the predicted coordinates for each point, after the deformation
    # (output dimensions: [B x 3 x num_points_to_decoder] )
    # point_code_size = 3 + global_code_size, is a code for each point

    def __init__(self, point_code_size=2500):
        self.point_code_size = point_code_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.point_code_size, self.point_code_size, 1)
        self.conv2 = torch.nn.Conv1d(self.point_code_size, self.point_code_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.point_code_size // 2, self.point_code_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.point_code_size // 4, self.point_code_size // 8, 1)
        self.conv5 = torch.nn.Conv1d(self.point_code_size // 8, self.point_code_size // 16, 1)
        self.conv6 = torch.nn.Conv1d(self.point_code_size // 16, self.point_code_size // 16, 1)
        self.conv7 = torch.nn.Conv1d(self.point_code_size // 16, self.point_code_size // 16, 1)
        self.conv8 = torch.nn.Conv1d(self.point_code_size // 16, self.point_code_size // 16, 1)
        self.conv9 = torch.nn.Conv1d(self.point_code_size // 16, 3, 1)  # OH: decoder output layer

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.point_code_size)
        self.bn2 = torch.nn.BatchNorm1d(self.point_code_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.point_code_size // 4)
        self.bn4 = torch.nn.BatchNorm1d(self.point_code_size // 8)
        self.bn5 = torch.nn.BatchNorm1d(self.point_code_size // 16)
        self.bn6 = torch.nn.BatchNorm1d(self.point_code_size // 16)
        self.bn7 = torch.nn.BatchNorm1d(self.point_code_size // 16)
        self.bn8 = torch.nn.BatchNorm1d(self.point_code_size // 16)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = 2 * self.th(self.conv9(x))
        return x


# OH:
class CompletionNet(nn.Module):  # OH: inherits from the base class - torch.nn.Module
    # OH: V
    def __init__(self, bottleneck_size=1024, num_input_channels = 3):
        super(CompletionNet, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.num_input_channels = num_input_channels

        # OH: The encoder takes a 3D point cloud as an input. Note that a linear layer is applied to the global
        # feature vector, as a final step in the encoder
        self.encoder = nn.Sequential(
            PointNetfeat(global_feat=True, trans=False, bottleneck_size=self.bottleneck_size, num_input_channels = self.num_input_channels),
            nn.Linear(self.bottleneck_size, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )

        # OH: The decoder takes as an input the template coordinates with the global feature vector of the input shape
        self.decoder = PointGenCon(point_code_size = num_input_channels + self.bottleneck_size + self.bottleneck_size)

    # OH: Takes as an input the partial point cloud and the template point cloud, encoding them, and decoding
    # the template deformation
    def forward(self, part, template):
        # OH: part, template [B x 3 x (num_points_part or num_points_template)]
        code_size = self.encoder[0].bottleneck_size
        batch_size = part.size(0)
        num_points_template = template.size(2)

        part_code = self.encoder(part)  # OH: [B x code_size]
        template_code = self.encoder(template)  # OH: [B x code_size]

        part_code = part_code.unsqueeze(2).expand(batch_size, code_size, num_points_template).contiguous()  # OH: [B x code_size x num_points_template]
        template_code = template_code.unsqueeze(2).expand(batch_size, code_size, num_points_template).contiguous()  # OH: [B x code_size x num_points_template]

        y = torch.cat((template, part_code, template_code), 1).contiguous()  # OH: [B x (3 + 2*code_size) x num_points_template]
        out = self.decoder(y).contiguous()
        # OH: [B x 9 x num_points_template]; first 3 channels represent axis location,
        # 3 next channels represent axis-angle vector, 3 last channels represent additional translation
        return out





