import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------------------------------
#                                               Decoders
# ----------------------------------------------------------------------------------------------------------------------
class ShapeDecoder(nn.Module):
    CCFG = [1, 1, 2, 4, 8, 16, 16]  # Enlarge this if you need more

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
                nn.init.normal_(m.weight, mean=bn_gamma_mu, std=bn_gamma_sigma)  # weight=gamma
                nn.init.constant_(m.bias, bn_betta_bias)  # bias=betta

    # noinspection PyTypeChecker
    # Input: Point code for each point: [b x nv x pnt_code_size]
    # Where pnt_code_size == in_channels + 2*shape_code
    # Output: predicted coordinates for each point, after the deformation [B x nv x 3]
    def forward(self, x):
        x = x.transpose(2, 1).contiguous()  # [b x nv x in_channels]
        for convl, bnl in zip(self.convls[:-1], self.bnls):
            x = F.relu(bnl(convl(x)))
        out = 2 * self.thl(self.convls[-1](x))  # TODO - Fix this constant - we need a global scale
        out = out.transpose(2, 1)
        return out


# ----------------------------------------------------------------------------------------------------------------------
#                                               Regressors
# ----------------------------------------------------------------------------------------------------------------------
class Regressor(nn.Module):
    # TODO: support external control on internal architecture
    def __init__(self, code_size):
        super().__init__()
        self.code_size = code_size

        self.lin1 = nn.Linear(2 * self.code_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.code_size)

        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(self.code_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.lin1(x)))
        x = F.relu(self.bn2(self.lin2(x)))
        x = F.relu(self.bn3(self.lin3(x)))
        return x


# ----------------------------------------------------------------------------------------------------------------------
#                                               Aux Classes
# ----------------------------------------------------------------------------------------------------------------------
class Template:
    def __init__(self, in_channels, dev):
        from cfg import UNIVERSAL_PRECISION, SMPL_TEMPLATE_PATH
        from util.mesh.io import read_ply
        from util.mesh.ops import batch_vnrmls

        vertices, faces, colors = read_ply(SMPL_TEMPLATE_PATH)
        self.vertices = torch.tensor(vertices, dtype=getattr(torch, UNIVERSAL_PRECISION)).unsqueeze(0)
        faces = torch.from_numpy(faces).long()
        self.in_channels = in_channels
        if self.in_channels == 6:
            self.normals, _ = batch_vnrmls(self.vertices, faces, return_f_areas=False)  # done on cpu (default)
            self.normals = self.normals.to(device=dev)  # transformed to requested device
        self.vertices = self.vertices.to(device=dev)  # transformed to requested device

        # self.colors = colors

    def get_template(self):
        if self.in_channels == 3:
            return self.vertices
        elif self.in_channels == 6:
            return torch.cat((self.vertices, self.normals), 2).contiguous()
        else:
            raise AssertionError