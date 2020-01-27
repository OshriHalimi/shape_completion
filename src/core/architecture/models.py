import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from test_tube import HyperOptArgumentParser
from architecture.lightning import CompletionLightningModel
from timeit import default_timer as timer

# ----------------------------------------------------------------------------------------------------------------------
#                                                      Full Models 
# ----------------------------------------------------------------------------------------------------------------------

class F2PEncoderDecoder(CompletionLightningModel):

    # @override
    def _build_model(self):
        # Encoder takes a 3D point cloud as an input. 
        # Note that a linear layer is applied to the global feature vector
        self.encoder = nn.Sequential(
            PointNetFeatures(only_global_feats=True, code_size=self.hparams.code_size,
                             in_channels=self.hparams.in_channels),
            nn.Linear(self.hparams.code_size, self.hparams.code_size),
            nn.BatchNorm1d(self.hparams.code_size),
            nn.ReLU()
        )
        self.decoder = CompletionDecoder(pnt_code_size=self.hparams.in_channels + 2 * self.hparams.code_size,
                                         out_channels=self.hparams.out_channels, num_convl=self.hparams.decoder_convl)

    # @override
    def _init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm1d):
                init.normal_(m.weight, mean=1.0, std=0.02)
                init.constant_(m.bias, 0)

    @staticmethod
    # TODO - Not sure if this placement is comfortable
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--code_size', default=1024, type=int)
        p.add_argument('--out_channels', default=3, type=int)
        p.add_argument('--decoder_convl', default=5, type=int)
        if not parent_parser:  # Name clash with parent
            p.add_argument('--in_channels', default=3, type=int)
        return p

    def forward(self, part, template):
        # start = timer()
        # TODO - Add handling of differently scaled meshes
        # part, template = b['gt_part'],b['tp']
        # part, template [bs x nv x 3]
        bs = part.size(0)
        nv = part.size(1)

        # TODO - Get rid of this transpose & contiguous
        part = part.transpose(2, 1).contiguous()
        template = template.transpose(2, 1).contiguous()
        # part, template [bs x 3 x nv]

        # [ b x code_size ] 
        part_code = self.encoder(part)
        template_code = self.encoder(template)

        # [b x code_size x nv]
        part_code = part_code.unsqueeze(2).expand(bs, self.hparams.code_size, nv)
        template_code = template_code.unsqueeze(2).expand(bs, self.hparams.code_size, nv)

        # [b x (3 + 2*code_size) x nv]
        y = torch.cat((template, part_code, template_code), 1).contiguous()
        y = self.decoder(y).transpose(2, 1)  # TODO - get rid of this transpose
        # end = timer()
        # print(end-start)
        return y


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
    # Input: Point code for each point: [b x pnt_code_size x nv]
    # Where pnt_code_size == 3 + 2*shape_code
    # Output: predicted coordinates for each point, after the deformation [B x 3 x nv]
    def __init__(self, pnt_code_size, out_channels, num_convl):
        self.point_code_size = pnt_code_size
        self.num_output_channels = out_channels
        super().__init__()
        self.conv1 = torch.nn.Conv1d(self.point_code_size, self.point_code_size, 1)
        self.conv2 = torch.nn.Conv1d(self.point_code_size, self.point_code_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.point_code_size // 2, self.point_code_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.point_code_size // 4, self.point_code_size // 8, 1)
        self.conv5 = torch.nn.Conv1d(self.point_code_size // 8, self.point_code_size // 16, 1)
        self.conv6 = torch.nn.Conv1d(self.point_code_size // 16, self.point_code_size // 16, 1)
        self.conv7 = torch.nn.Conv1d(self.point_code_size // 16, self.point_code_size // 16, 1)
        self.conv8 = torch.nn.Conv1d(self.point_code_size // 16, self.point_code_size // 16, 1)
        self.conv9 = torch.nn.Conv1d(self.point_code_size // 16, self.num_output_channels, 1)  # OH: decoder output layer

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


# ----------------------------------------------------------------------------------------------------------------------
#                                                          Graveyard
# ----------------------------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    pass
