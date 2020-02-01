from architecture.lightning import CompletionLightningModel
from test_tube import HyperOptArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------------------------------
#                                                      Full Models 
# ----------------------------------------------------------------------------------------------------------------------

class F2PEncoderDecoder(CompletionLightningModel):

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

    def _init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=self.hparams.init_conv_mu, std=self.hparams.init_conv_sigma)
                # TODO - What about bias ? Is it 0 ?
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, mean=self.hparams.init_batch_norm_mu, std=self.hparams.init_batch_norm_sigma)
                nn.init.constant_(m.bias, self.hparams.init_batch_norm_bias)

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--code_size', default=1024, type=int)
        p.add_argument('--out_channels', default=3, type=int)
        p.add_argument('--decoder_convl', default=5, type=int)

        # Init:
        p.add_argument('--init_conv_mu', default=0, type=float)
        p.add_argument('--init_conv_sigma', default=0.02, type=float)

        p.add_argument('--init_batch_norm_mu', default=1, type=float)
        p.add_argument('--init_batch_norm_sigma', default=0.02, type=float)
        p.add_argument('--init_batch_norm_bias', default=0, type=float)

        if not parent_parser:  # Name clash with parent
            p.add_argument('--in_channels', default=3, type=int)
        return p

    def forward(self, part, template):
        # TODO - Add handling of differently scaled meshes
        # part, template [bs x nv x 3]
        bs = part.size(0)
        nv = part.size(1)

        # TODO - Get rid of this transpose & contiguous
        part = part.transpose(2, 1).contiguous()  # [bs x 3 x nv]
        template = template.transpose(2, 1).contiguous()  # [bs x 3 x nv]

        part_code = self.encoder(part)  # [ b x code_size ]
        template_code = self.encoder(template)  # [ b x code_size ]

        part_code = part_code.unsqueeze(2).expand(bs, self.hparams.code_size, nv)  # [b x code_size x nv]
        template_code = template_code.unsqueeze(2).expand(bs, self.hparams.code_size, nv)  # [b x code_size x nv]

        y = torch.cat((template, part_code, template_code), 1).contiguous()  # [b x (3 + 2*code_size) x nv]
        y = self.decoder(y).transpose(2, 1)  # TODO - get rid of this transpose
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

    # noinspection PyTypeChecker
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
        self.convls = nn.ModuleList(self.convls)
        self.bnls = nn.ModuleList(self.bnls)

    # noinspection PyTypeChecker
    def forward(self, x):

        for convl, bnl in zip(self.convls[:-1], self.bnls):
            x = F.relu(bnl(convl(x)))
        return 2 * self.thl(self.convls[-1](x))  # TODO - Fix this constant - we need a global scale
