import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from architecture.PytorchNet import PytorchNet
import torch.nn.init as init


# ----------------------------------------------------------------------------------------------------------------------
#                                                      Full Models 
# ----------------------------------------------------------------------------------------------------------------------

class CompletionNet(PytorchNet):
    def __init__(self, code_size=1024, in_channels=3, out_channels=3, decoder_convl=5):
        super().__init__()
        self.code_size = code_size  # Also the code size 
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder takes a 3D point cloud as an input. 
        # Note that a linear layer is applied to the global feature vector
        self.encoder = nn.Sequential(
            PointNetFeatures(only_global_feats=True, code_size=self.code_size, in_channels=self.in_channels),
            nn.Linear(self.code_size, self.code_size),
            nn.BatchNorm1d(self.code_size),
            nn.ReLU()
        )
        self.decoder = CompletionDecoder(pnt_code_size=in_channels + 2 * self.code_size,
                                         out_channels=self.out_channels, num_convl=decoder_convl)

        self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.normal_(m.weight, mean=1.0, std=0.02)
            elif isinstance(m, nn.BatchNorm1d):
                init.normal_(m.weight, mean=1.0, std=0.02)
                init.constant_(m.bias, 0)

    def forward(self, part, template):
        # TODO - Get rid of this transpose & contiguous
        # part, template [b x nv x 3]
        part = part.transpose(2, 1).contiguous()
        template = template.transpose(2, 1).contiguous()
        # part, template [b x 3 x nv]
        b = part.size(0)
        nv = part.size(2)

        # [ b x code_size ] 
        part_code = self.encoder(part)
        template_code = self.encoder(template)

        # [b x code_size x nv]
        part_code = part_code.unsqueeze(2).expand(b, self.code_size, nv)
        template_code = template_code.unsqueeze(2).expand(b, self.code_size, nv)

        # [b x (3 + 2*code_size) x nv]
        y = torch.cat((template, part_code, template_code), 1).contiguous()
        return self.decoder(y).transpose(2, 1)  # TODO - get rid of this transpose


# ----------------------------------------------------------------------------------------------------------------------
#                                               Sub Models 
# ----------------------------------------------------------------------------------------------------------------------

class PointNetFeatures(nn.Module):
    # Input: Batch of Point Clouds : [b x 3 x num_vertices] 
    # Output: 
    # If concat_local_feats = False: The global feature vector : [b x code_size]
    # Else: The point features & global vector [ b x (64+code_size) x num_vertices ]
    def __init__(self, only_global_feats=True, code_size=1024, in_channels=3):
        # If only_global_feats==True, returns a single global feature each point cloud;
        # Else: the global feature vector is copied for each point and concatenated with the point features;
        # In this case, the final structure is similar to the initial input of the segmentation network in PointNet
        # See PointNet diagram: https://arxiv.org/pdf/1612.00593.pdf)
        super().__init__()
        self.code_size = code_size
        self.in_channels = in_channels
        self.only_global_feats = only_global_feats

        self.conv1 = nn.Conv1d(self.in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, self.code_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.code_size)

        # self.trans = trans
        # self.stn = STN3d()
        # if self.trans:
        #    trans = self.stn(x)  # OH: batch transformation; [B x 3 x 3].
        #    # OH: In practice the point cloud is rotated by the transpose of trans
        #    x = x.transpose(2, 1)  # OH: [B x 3 x n] --> [B x n x 3]
        #    x = torch.bmm(x, trans)  # OH: batch matrix-matrix product
        #    x = x.transpose(2, 1).contiguous()  # OH: [B x n x 3] --> [B x 3 x n]

    def forward(self, x):

        nv = x.shape[2]
        x = F.relu(self.bn1(self.conv1(x)))
        if not self.only_global_feats:  # Save GPU Memory
            point_feats = x  # OH: [B x 64 x nv]
        x = F.relu(self.bn2(self.conv2(x)))  # [B x 128 x n]
        x = self.bn3(self.conv3(x))  # [B x code_size x n]
        x, _ = torch.max(x, 2)  # [B x code_size]
        x = x.view(-1, self.code_size)  # OH: redundant line ???
        if self.only_global_feats:
            return x
        else:
            x = x.view(-1, self.code_size, 1).repeat(1, 1, nv)
            return torch.cat([x, point_feats], 1)


class CompletionDecoder(nn.Module):
    CCFG = [1, 1, 2, 4, 8, 16, 16]  # Enlarge this if you need more

    # Input: Point code for each point: [b x pnt_code_size x nv]
    # Where pnt_code_size == 3 + 2*shape_code
    # Output: predicted coordinates for each point, after the deformation [B x 3 x nv]
    def __init__(self, pnt_code_size, out_channels, num_convl):
        super().__init__()

        self.pnt_code_size = pnt_code_size
        self.out_channels = out_channels
        if num_convl > len(self.CCFG):
            raise NotImplementedError("Please enlarge the Conv Config vector")

        self.thl = nn.Tanh()
        self.convls = []
        self.bnls = []
        for i in range(num_convl - 1):
            self.convls.append(nn.Conv1d(self.pnt_code_size // self.CCFG[i], self.pnt_code_size // self.CCFG[i + 1], 1))
            self.bnls.append(nn.BatchNorm1d(self.pnt_code_size // self.CCFG[i + 1]))
        self.convls.append(nn.Conv1d(self.pnt_code_size // self.CCFG[num_convl - 1], self.out_channels, 1))

    def forward(self, x):

        for convl, bnl in zip(self.convls[:-1], self.bnls):
            x = F.relu(bnl(convl(x)))
        return 2 * self.thl(self.convls[-1](x))  # TODO - Why is there a 2 here?


# ----------------------------------------------------------------------------------------------------------------------
#                                                          Graveyard
# ----------------------------------------------------------------------------------------------------------------------
class STN3d(nn.Module):  # OH: An alignment network T-Net (Probably)
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

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
        r = torch.eye(3).expand(x.shape[0], 3, 3).clone()
        r[:, 0, 0] = torch.cos(x)
        r[:, 0, 2] = -torch.sin(x)
        r[:, 2, 0] = torch.sin(x)
        r[:, 2, 2] = torch.cos(x)
        r = r.contiguous().cuda()
        return r


if __name__ == "__main__":
    pass
