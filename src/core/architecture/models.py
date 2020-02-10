from architecture.lightning import CompletionLightningModel
from test_tube import HyperOptArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
from mesh.io import read_ply
from cfg import SMPL_TEMPLATE_PATH
from mesh.ops import batch_vnrmls


# ----------------------------------------------------------------------------------------------------------------------
#                                                      Full Networks
# ----------------------------------------------------------------------------------------------------------------------
class F2PEncoderDecoder(CompletionLightningModel):
    def _build_model(self):
        # Encoder takes a 3D point cloud as an input. 
        # Note that a linear layer is applied to the global feature vector
        self.encoder = ShapeEncoder(in_channels=self.hparams.in_channels, code_size=self.hparams.code_size)
        self.decoder = ShapeDecoder(pnt_code_size=self.hparams.in_channels + 2 * self.hparams.code_size,
                                         out_channels=self.hparams.out_channels, num_convl=self.hparams.decoder_convl)

    def _init_model(self):
        self.encoder.init_weights()
        self.decoder.init_weights()

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--code_size', default=1024, type=int)
        p.add_argument('--out_channels', default=3, type=int)
        p.add_argument('--decoder_convl', default=5, type=int)
        if not parent_parser:  # Name clash with parent
            p.add_argument('--in_channels', default=3, type=int)
        return p

    def forward(self, input_dict):
        part = input_dict['gt_part']
        full = input_dict['tp']

        # part, full [bs x nv x in_channels]
        bs = part.size(0)
        nv = part.size(1)

        part_code = self.encoder(part)  # [b x code_size]
        full_code = self.encoder(full)  # [b x code_size]

        part_code = part_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]
        full_code = full_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]

        y = torch.cat((full, part_code, full_code), 2).contiguous()  # [b x nv x (in_channels + 2*code_size)]
        y = self.decoder(y)
        return {'completion': y}


class F2PEncoderDecoderSkeptic(CompletionLightningModel):
    def _build_model(self):
        # Encoder takes a 3D point cloud as an input.
        # Note that a linear layer is applied to the global feature vector
        vertices, faces, colors = read_ply(SMPL_TEMPLATE_PATH)
        self.template = Template(vertices, faces, colors, self.hparams.in_channels, self.hparams.dev)
        self.encoder = ShapeEncoder(in_channels=self.hparams.in_channels, code_size=self.hparams.code_size, dense=self.hparams.dense_encoder)
        self.comp_decoder = ShapeDecoder(pnt_code_size=self.hparams.in_channels + 2 * self.hparams.code_size,
                                         out_channels=self.hparams.out_channels, num_convl=self.hparams.comp_decoder_convl)
        self.rec_decoder = ShapeDecoder(pnt_code_size=self.hparams.in_channels + self.hparams.code_size,
                                         out_channels=self.hparams.out_channels, num_convl=self.hparams.rec_decoder_convl)


    def _init_model(self):
        self.encoder.init_weights()
        self.comp_decoder.init_weights()
        self.rec_decoder.init_weights()

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--code_size', default=1024, type=int)
        p.add_argument('--out_channels', default=3, type=int)
        p.add_argument('--comp_decoder_convl', default=5, type=int)
        p.add_argument('--rec_decoder_convl', default=3, type=int)
        if not parent_parser:  # Name clash with parent
            p.add_argument('--in_channels', default=3, type=int)
        return p

    def forward(self, input_dict):
        part = input_dict['gt_part']
        full = input_dict['tp']
        # part, full [bs x nv x in_channels]
        bs = part.size(0)
        nv = part.size(1)

        part_code = self.encoder(part)  # [b x code_size]
        full_code = self.encoder(full)  # [b x code_size]

        part_code = part_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]
        full_code = full_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]

        x = torch.cat((full, part_code, full_code), 2).contiguous()  # [b x nv x (in_channels + 2*code_size)]
        completion = self.comp_decoder(x)

        template = self.template.get_template().expand(bs, nv, self.hparams.in_channels)
        y = torch.cat((template, full_code), 2).contiguous()  # [b x nv x (in_channels + code_size)]
        full_rec = self.rec_decoder(y)

        z = torch.cat((template, part_code), 2).contiguous()  # [b x nv x (in_channels + code_size)]
        part_rec = self.rec_decoder(z)

        return {'completion': completion, 'full_rec':full_rec, 'part_rec':part_rec}


class F2PEncoderDecoderVerySkeptic(CompletionLightningModel):
    def _build_model(self):
        # Encoder takes a 3D point cloud as an input.
        # Note that a linear layer is applied to the global feature vector
        vertices, faces, colors = read_ply(SMPL_TEMPLATE_PATH)
        self.template = Template(vertices, faces, colors, self.hparams.in_channels, self.hparams.dev)
        self.encoder = ShapeEncoder(in_channels=self.hparams.in_channels, code_size=self.hparams.code_size, dense=self.hparams.dense_encoder)
        self.decoder = ShapeDecoder(pnt_code_size=self.hparams.in_channels + self.hparams.code_size,
                                         out_channels=self.hparams.out_channels, num_convl=self.hparams.decoder_convl)
        self.regressor = Regressor(code_size=self.hparams.code_size)


    def _init_model(self):
        self.encoder.init_weights()
        self.decoder.init_weights()
        #TODO: (optionally) non default weight init of regressor

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--code_size', default=1024, type=int)
        p.add_argument('--out_channels', default=3, type=int)
        p.add_argument('--decoder_convl', default=5, type=int)
        if not parent_parser:  # Name clash with parent
            p.add_argument('--in_channels', default=3, type=int)
        return p

    def forward(self, input_dict):
        part = input_dict['gt_part']
        full = input_dict['tp']
        gt = input_dict['gt']

        # part, full, gt [bs x nv x in_channels]
        bs = part.size(0)
        nv = part.size(1)

        part_code = self.encoder(part)  # [b x code_size]
        full_code = self.encoder(full)  # [b x code_size]
        gt_code = self.encoder(gt)  # [b x code_size]
        comp_code = self.regressor(torch.cat((part_code, full_code), 1).contiguous())
        output_dict = {'comp_code': comp_code, 'gt_code': gt_code}

        part_code = part_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]
        full_code = full_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]
        comp_code = comp_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]

        template = self.template.get_template().expand(bs, nv, self.hparams.in_channels)
        full_rec = self.decoder(torch.cat((template, full_code), 2).contiguous()) # decoder input: [b x nv x (in_channels + code_size)]
        part_rec = self.decoder(torch.cat((template, part_code), 2).contiguous()) # decoder input: [b x nv x (in_channels + code_size)]
        completion = self.decoder(torch.cat((template, comp_code), 2).contiguous()) # decoder input: [b x nv x (in_channels + code_size)]

        output_dict.update({'completion': completion, 'full_rec':full_rec, 'part_rec':part_rec})
        return output_dict
# ----------------------------------------------------------------------------------------------------------------------
#                                               Encoders
# ----------------------------------------------------------------------------------------------------------------------
class ShapeEncoder(nn.Module):
    def __init__(self, code_size=1024, in_channels=3, dense=True):
        super().__init__()
        self.code_size = code_size
        self.in_channels = in_channels

        if dense:
            FeaturesNet = DensePointNetFeatures(self.code_size, self.in_channels)
        else:
            FeaturesNet = PointNetFeatures(self.code_size, self.in_channels)

        self.encoder = nn.Sequential(
            FeaturesNet,
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
    #TODO: support external control on internal architecture
    def __init__(self,code_size):
        super().__init__()
        self.code_size = code_size

        self.lin1 = nn.Linear(2 * self.code_size,  128)
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
#                                               Modules
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
        x = x.transpose(2, 1).contiguous()  #[b x in_channels x num_vertices]
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
        x = x.transpose(2, 1).contiguous()  #[b x in_channels x num_vertices]
        y = F.relu(self.bn1(self.conv1(x)))  # [B x 64 x n]
        z = F.relu(self.bn2(self.conv2(torch.cat((x,y), 1))))  # [B x 128 x n]
        z = self.bn3(self.conv3(torch.cat((x,y,z), 1)))  # [B x code_size x n]
        z, _ = torch.max(z, 2)  # [B x code_size]
        return z
# ----------------------------------------------------------------------------------------------------------------------
#                                               Aux Classes
# ----------------------------------------------------------------------------------------------------------------------
class Template():
    def __init__(self, vertices, faces, colors, in_channels, dev):
        from cfg import UNIVERSAL_PRECISION
        self.vertices = torch.tensor(vertices, dtype=getattr(torch,UNIVERSAL_PRECISION)).unsqueeze(0)
        faces = torch.LongTensor(faces)  # Not a property
        self.in_channels = in_channels
        if self.in_channels == 6:
            self.normals, _ = batch_vnrmls(self.vertices, faces, return_f_areas=False)  #done on cpu (default)
            self.normals = self.normals.to(device = dev)  #transformed to requested device
        self.vertices = self.vertices.to(device = dev) #transformed to requested device

        #self.colors = colors


    def get_template(self):
        if self.in_channels == 3:
            return self.vertices
        elif self.in_channels == 6:
            return torch.cat((self.vertices, self.normals), 2).contiguous()
        else:
            raise AssertionError


