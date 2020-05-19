from lightning.completion_net import CompletionLightningModel
from test_tube import HyperOptArgumentParser
import torch
from architecture.encoders import ShapeEncoder, ShapeEncoderDGCNN
from architecture.decoders import ShapeDecoder, Template, Regressor
from cfg import UNIVERSAL_PRECISION

# ----------------------------------------------------------------------------------------------------------------------
#                                                      BASE
# ----------------------------------------------------------------------------------------------------------------------
class F2PEncoderDecoder(CompletionLightningModel):
    def _build_model(self):
        # Encoder takes a 3D point cloud as an input.
        # Note that a linear layer is applied to the global feature vector
        self.encoder = ShapeEncoder(in_channels=self.hparams.in_channels, code_size=self.hparams.code_size,
                                    dense=self.hparams.dense_encoder)
        self.decoder = ShapeDecoder(pnt_code_size=self.hparams.code_size_in_shape + 2 * self.hparams.code_size, #self.hparams.in_channels --> 100 latent space dimension
                                    out_channels=self.hparams.out_channels, num_convl=self.hparams.decoder_convl)

    def _init_model(self):
        self.encoder.init_weights()
        self.decoder.init_weights()

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--dense_encoder', default=False, type=bool)
        p.add_argument('--code_size', default=512, type=int)
        p.add_argument('--code_size_in_shape', default=30, type=int)
        p.add_argument('--out_channels', default=3, type=int)
        p.add_argument('--decoder_convl', default=5, type=int)
        if not parent_parser:  # Name clash with parent
            p.add_argument('--in_channels', default=3, type=int)
        return p

    def forward(self, input_dict):
        part = input_dict['gt_part'] if 'gt_part' in input_dict else input_dict['gt_noise'] # TODO - Generalize this
        full = input_dict['tp']

        # part, full [bs x nv x in_channels]
        bs = part.size(0)
        nv = part.size(1)

        part_code = self.encoder(part)  # [b x code_size]
        full_code = self.encoder(full)  # [b x code_size]

        part_code = part_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]
        full_code = full_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]

        latent_in_shape = torch.randn(bs, nv, self.hparams.code_size_in_shape, device=self.hparams.dev, dtype=getattr(torch, UNIVERSAL_PRECISION), requires_grad = False)
        #y = torch.cat((full, part_code, full_code), 2).contiguous()  # [b x nv x (in_channels + 2*code_size)]
        y = torch.cat((latent_in_shape, part_code, full_code), 2).contiguous()  # [b x nv x (code_size_in_shape + 2*code_size)]
        y = self.decoder(y)
        return {'completion_xyz': y}


# ----------------------------------------------------------------------------------------------------------------------
#                                                      BASE
# ----------------------------------------------------------------------------------------------------------------------
class F2PEncoderDecoderRealistic(CompletionLightningModel):
    def _build_model(self):
        # Encoder takes a 3D point cloud as an input.
        # Note that a linear layer is applied to the global feature vector
        self.encoder_full = ShapeEncoder(in_channels=self.hparams.in_channels, code_size=self.hparams.code_size,
                                         dense=self.hparams.dense_encoder)
        self.encoder_part = ShapeEncoder(in_channels=3, code_size=self.hparams.code_size,
                                         dense=self.hparams.dense_encoder)
        self.decoder = ShapeDecoder(pnt_code_size=self.hparams.in_channels + 2 * self.hparams.code_size,
                                    out_channels=self.hparams.out_channels, num_convl=self.hparams.decoder_convl)

    def _init_model(self):
        self.encoder_full.init_weights()
        self.encoder_part.init_weights()
        self.decoder.init_weights()

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--dense_encoder', default=False, type=bool)
        p.add_argument('--code_size', default=512, type=int)
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

        part_code = self.encoder_part(part[:, :, :3])  # [b x code_size]
        full_code = self.encoder_full(full)  # [b x code_size]

        part_code = part_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]
        full_code = full_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]

        y = torch.cat((full, part_code, full_code), 2).contiguous()  # [b x nv x (in_channels + 2*code_size)]
        y = self.decoder(y)
        return {'completion_xyz': y}


class F2PDGCNNEncoderDecoder(F2PEncoderDecoder):
    def _build_model(self):
        self.encoder = ShapeEncoderDGCNN(in_channels=self.hparams.in_channels, code_size=self.hparams.code_size,
                                         k=20, device=self.hparams.dev)
        self.decoder = ShapeDecoder(pnt_code_size=self.hparams.in_channels + 2 * self.hparams.code_size,
                                    out_channels=self.hparams.out_channels, num_convl=self.hparams.decoder_convl)


# ----------------------------------------------------------------------------------------------------------------------
#                                                      BASE
# ----------------------------------------------------------------------------------------------------------------------
class F2PEncoderDecoderSkeptic(CompletionLightningModel):
    def _build_model(self):
        # Encoder takes a 3D point cloud as an input.
        # Note that a linear layer is applied to the global feature vector
        self.template = Template(self.hparams.in_channels, self.hparams.dev)
        self.encoder = ShapeEncoder(in_channels=self.hparams.in_channels, code_size=self.hparams.code_size,
                                    dense=self.hparams.dense_encoder)
        self.comp_decoder = ShapeDecoder(pnt_code_size=self.hparams.in_channels + 2 * self.hparams.code_size,
                                         out_channels=self.hparams.out_channels,
                                         num_convl=self.hparams.comp_decoder_convl)
        self.rec_decoder = ShapeDecoder(pnt_code_size=self.hparams.in_channels + self.hparams.code_size,
                                        out_channels=self.hparams.out_channels,
                                        num_convl=self.hparams.rec_decoder_convl)

    def _init_model(self):
        self.encoder.init_weights()
        self.comp_decoder.init_weights()
        self.rec_decoder.init_weights()

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--dense_encoder', default=False, type=bool)
        p.add_argument('--code_size', default=512, type=int)
        p.add_argument('--out_channels', default=3, type=int)
        p.add_argument('--comp_decoder_convl', default=5, type=int)
        p.add_argument('--rec_decoder_convl', default=3, type=int)
        if not parent_parser:  # Name clash with parent
            p.add_argument('--in_channels', default=3, type=int)
        return p

    def forward(self, input_dict):
        part = input_dict['gt_part']
        full = input_dict['tp']
        gt = input_dict['gt']
        # part, full [bs x nv x in_channels]
        bs = part.size(0)
        nv = part.size(1)

        part_code = self.encoder(part)  # [b x code_size]
        full_code = self.encoder(full)  # [b x code_size]
        gt_code = self.encoder(gt)  # [b x code_size]

        part_code = part_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]
        full_code = full_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]
        gt_code = gt_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]

        completion = self.comp_decoder(torch.cat((full, part_code, full_code), 2).contiguous())

        template = self.template.get_template().expand(bs, nv, self.hparams.in_channels)
        full_rec = self.rec_decoder(torch.cat((template, full_code), 2).contiguous())
        part_rec = self.rec_decoder(torch.cat((template, part_code), 2).contiguous())
        gt_rec = self.rec_decoder(torch.cat((template, gt_code), 2).contiguous())

        return {'completion_xyz': completion, 'full_rec': full_rec, 'part_rec': part_rec, 'gt_rec': gt_rec}


# ----------------------------------------------------------------------------------------------------------------------
#                                                      BASE
# ----------------------------------------------------------------------------------------------------------------------
class F2PEncoderJointDecoderSkeptic(CompletionLightningModel):
    def _build_model(self):
        # Encoder takes a 3D point cloud as an input.
        # Note that a linear layer is applied to the global feature vector
        self.template = Template(self.hparams.in_channels, self.hparams.dev)
        self.encoder = ShapeEncoder(in_channels=self.hparams.in_channels, code_size=self.hparams.code_size,
                                    dense=self.hparams.dense_encoder)
        self.comp_decoder = ShapeDecoder(pnt_code_size=self.hparams.in_channels + 2 * self.hparams.code_size,
                                         out_channels=self.hparams.out_channels,
                                         num_convl=self.hparams.comp_decoder_convl)
        self.rec_decoder = ShapeDecoder(pnt_code_size=self.hparams.in_channels + self.hparams.code_size,
                                        out_channels=self.hparams.out_channels,
                                        num_convl=self.hparams.rec_decoder_convl)

    def _init_model(self):
        self.encoder.init_weights()
        self.comp_decoder.init_weights()
        self.rec_decoder.init_weights()

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--dense_encoder', default=False, type=bool)
        p.add_argument('--code_size', default=512, type=int)
        p.add_argument('--out_channels', default=3, type=int)
        p.add_argument('--comp_decoder_convl', default=5, type=int)
        p.add_argument('--rec_decoder_convl', default=3, type=int)
        if not parent_parser:  # Name clash with parent
            p.add_argument('--in_channels', default=3, type=int)
        return p

    def forward(self, input_dict):
        part = input_dict['gt_part']
        full_1 = input_dict['tp1']
        full_2 = input_dict['tp2']
        gt = input_dict['gt']
        # part, full [bs x nv x in_channels]
        bs = part.size(0)
        nv = part.size(1)

        part_code = self.encoder(part)  # [b x code_size]
        full_code_1 = self.encoder(full_1)  # [b x code_size]
        full_code_2 = self.encoder(full_2)  # [b x code_size]
        gt_code = self.encoder(gt)  # [b x code_size]

        part_code = part_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]
        full_code_1 = full_code_1.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]
        full_code_2 = full_code_2.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]
        gt_code = gt_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]

        completion_1 = self.comp_decoder(torch.cat((full_1, part_code, full_code_1), 2).contiguous())
        completion_2 = self.comp_decoder(torch.cat((full_2, part_code, full_code_2), 2).contiguous())

        template = self.template.get_template().expand(bs, nv, self.hparams.in_channels)
        full_rec_1 = self.rec_decoder(torch.cat((template, full_code_1), 2).contiguous())
        full_rec_2 = self.rec_decoder(torch.cat((template, full_code_2), 2).contiguous())
        part_rec = self.rec_decoder(torch.cat((template, part_code), 2).contiguous())
        gt_rec = self.rec_decoder(torch.cat((template, gt_code), 2).contiguous())

        # TODO: What if the ntetwork returns more than one completion?
        return {'completion_xyz': completion_1, 'completion_xyz_2': completion_2, 'full_rec': full_rec_1,
                'full_rec_2': full_rec_2, 'part_rec': part_rec, 'gt_rec': gt_rec}


# ----------------------------------------------------------------------------------------------------------------------
#                                                      BASE
# ----------------------------------------------------------------------------------------------------------------------
class F2PEncoderRegressorDecoderSkeptic(CompletionLightningModel):
    def _build_model(self):
        # Encoder takes a 3D point cloud as an input.
        # Note that a linear layer is applied to the global feature vector
        self.template = Template(self.hparams.in_channels, self.hparams.dev)
        self.encoder = ShapeEncoder(in_channels=self.hparams.in_channels, code_size=self.hparams.code_size,
                                    dense=self.hparams.dense_encoder)
        self.comp_decoder = ShapeDecoder(pnt_code_size=self.hparams.in_channels + self.hparams.code_size,
                                         out_channels=self.hparams.out_channels,
                                         num_convl=self.hparams.comp_decoder_convl)
        self.rec_decoder = ShapeDecoder(pnt_code_size=self.hparams.in_channels + self.hparams.code_size,
                                        out_channels=self.hparams.out_channels,
                                        num_convl=self.hparams.rec_decoder_convl)
        self.regressor = Regressor(code_size=self.hparams.code_size)

    def _init_model(self):
        self.encoder.init_weights()
        self.rec_decoder.init_weights()
        self.comp_decoder.init_weights()
        # TODO: (optionally) non default weight init of regressor

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--dense_encoder', default=False, type=bool)
        p.add_argument('--code_size', default=512, type=int)
        p.add_argument('--out_channels', default=3, type=int)
        p.add_argument('--comp_decoder_convl', default=3, type=int)
        p.add_argument('--rec_decoder_convl', default=3, type=int)
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
        gt_code = gt_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]

        completion = self.comp_decoder(
            torch.cat((full, comp_code), 2).contiguous())  # decoder input: [b x nv x (in_channels + code_size)]

        template = self.template.get_template().expand(bs, nv, self.hparams.in_channels)
        full_rec = self.rec_decoder(
            torch.cat((template, full_code), 2).contiguous())  # decoder input: [b x nv x (in_channels + code_size)]
        part_rec = self.rec_decoder(
            torch.cat((template, part_code), 2).contiguous())  # decoder input: [b x nv x (in_channels + code_size)]
        gt_rec = self.rec_decoder(
            torch.cat((template, gt_code), 2).contiguous())  # decoder input: [b x nv x (in_channels + code_size)]

        output_dict.update({'completion_xyz': completion, 'full_rec': full_rec, 'part_rec': part_rec, 'gt_rec': gt_rec})
        return output_dict


# ----------------------------------------------------------------------------------------------------------------------
#                                                      BASE
# ----------------------------------------------------------------------------------------------------------------------
class PointContextNet(CompletionLightningModel):
    def _build_model(self):
        # Encoder takes a 3D point cloud as an input.
        # Note that a linear layer is applied to the global feature vector
        self.template = Template(self.hparams.in_channels, self.hparams.dev)
        self.encoder = ShapeEncoder(in_channels=self.hparams.in_channels, code_size=self.hparams.code_size,
                                    dense=self.hparams.dense_encoder)
        self.rec_decoder = ShapeDecoder(pnt_code_size=self.hparams.in_channels + self.hparams.code_size,
                                        out_channels=self.hparams.out_channels,
                                        num_convl=self.hparams.rec_decoder_convl)
        self.comp_decoder = ShapeDecoder(
            pnt_code_size=self.hparams.in_channels + self.hparams.out_channels + self.hparams.code_size * 2,
            out_channels=self.hparams.out_channels, num_convl=self.hparams.comp_decoder_convl)
        self.context_decoder = ShapeDecoder(pnt_code_size=self.hparams.in_channels + self.hparams.code_size,
                                            out_channels=self.hparams.out_channels,
                                            num_convl=self.hparams.context_decoder_convl)

    def _init_model(self):
        self.encoder.init_weights()
        self.rec_decoder.init_weights()
        self.comp_decoder.init_weights()
        self.context_decoder.init_weights()
        # TODO: (optionally) non default weight init of regressor

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--dense_encoder', default=False, type=bool)
        p.add_argument('--code_size', default=512, type=int)
        p.add_argument('--out_channels', default=3, type=int)
        p.add_argument('--rec_decoder_convl', default=3, type=int)
        p.add_argument('--comp_decoder_convl', default=3, type=int)
        p.add_argument('--context_decoder_convl', default=3, type=int)
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
        comp_code = torch.cat((part_code, full_code), 1).contiguous()  # [b x 2*code_size]

        part_code = part_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]
        full_code = full_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]
        comp_code = comp_code.unsqueeze(1).expand(bs, nv, 2 * self.hparams.code_size)  # [b x nv x 2*code_size]
        gt_code = gt_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]

        # template reconsturction
        template = self.template.get_template().expand(bs, nv, self.hparams.in_channels)
        full_rec = self.rec_decoder(
            torch.cat((template, full_code), 2).contiguous())  # decoder input: [b x nv x (in_channels + code_size)]
        part_rec = self.rec_decoder(
            torch.cat((template, part_code), 2).contiguous())  # decoder input: [b x nv x (in_channels + code_size)]
        gt_rec = self.rec_decoder(
            torch.cat((template, gt_code), 2).contiguous())  # decoder input: [b x nv x (in_channels + code_size)]

        # Point context $ completion
        point_context = self.context_decoder(
            torch.cat((full, full_code), 2).contiguous())  # decoder input: [b x nv x (in_channels + code_size)]
        completion = self.comp_decoder(torch.cat((full, point_context, comp_code),
                                                 2).contiguous())  # decoder input: [b x nv x (2 * in_channels + 2 * code_size)]

        output_dict = {'completion_xyz': completion, 'point_context': point_context, 'full_rec': full_rec,
                       'part_rec': part_rec, 'gt_rec': gt_rec}
        return output_dict


# ----------------------------------------------------------------------------------------------------------------------
#                                                      BASE
# ----------------------------------------------------------------------------------------------------------------------
class F2PEncoderDecoderTBased(CompletionLightningModel):
    def _build_model(self):
        # Encoder takes a 3D point cloud as an input.
        # Note that a linear layer is applied to the global feature vector
        self.template = Template(self.hparams.in_channels, self.hparams.dev)
        self.encoder = ShapeEncoder(in_channels=self.hparams.in_channels, code_size=self.hparams.code_size,
                                    dense=self.hparams.dense_encoder)
        self.decoder = ShapeDecoder(pnt_code_size=self.hparams.in_channels + self.hparams.code_size,
                                    out_channels=self.hparams.out_channels, num_convl=self.hparams.decoder_convl)
        self.regressor = Regressor(code_size=self.hparams.code_size)

    def _init_model(self):
        self.encoder.init_weights()
        self.decoder.init_weights()
        # TODO: (optionally) non default weight init of regressor

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--dense_encoder', default=False, type=bool)
        p.add_argument('--code_size', default=512, type=int)
        p.add_argument('--out_channels', default=3, type=int)
        p.add_argument('--decoder_convl', default=3, type=int)
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
        gt_code = gt_code.unsqueeze(1).expand(bs, nv, self.hparams.code_size)  # [b x nv x code_size]

        # all reconsturction (also completion are achieved by FIXED template deformation)
        template = self.template.get_template().expand(bs, nv, self.hparams.in_channels)
        full_rec = self.decoder(
            torch.cat((template, full_code), 2).contiguous())  # decoder input: [b x nv x (in_channels + code_size)]
        part_rec = self.decoder(
            torch.cat((template, part_code), 2).contiguous())  # decoder input: [b x nv x (in_channels + code_size)]
        gt_rec = self.decoder(
            torch.cat((template, gt_code), 2).contiguous())  # decoder input: [b x nv x (in_channels + code_size)]
        completion = self.decoder(
            torch.cat((template, comp_code), 2).contiguous())  # decoder input: [b x nv x (in_channels + code_size)]

        output_dict.update({'completion_xyz': completion, 'full_rec': full_rec, 'part_rec': part_rec, 'gt_rec': gt_rec})
        return output_dict
