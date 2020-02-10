from architecture.lightning import CompletionLightningModel
from test_tube import HyperOptArgumentParser
import torch
from architecture.encoders import ShapeEncoder
from architecture.decoders import ShapeDecoder, Template, Regressor


# ----------------------------------------------------------------------------------------------------------------------
#                                                      BASE
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
        return {'completion_xyz': y}


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

        return {'completion_xyz': completion, 'full_rec': full_rec, 'part_rec': part_rec}


# ----------------------------------------------------------------------------------------------------------------------
#                                                      BASE
# ----------------------------------------------------------------------------------------------------------------------
class F2PEncoderDecoderVerySkeptic(CompletionLightningModel):
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
        full_rec = self.decoder(
            torch.cat((template, full_code), 2).contiguous())  # decoder input: [b x nv x (in_channels + code_size)]
        part_rec = self.decoder(
            torch.cat((template, part_code), 2).contiguous())  # decoder input: [b x nv x (in_channels + code_size)]
        completion = self.decoder(
            torch.cat((template, comp_code), 2).contiguous())  # decoder input: [b x nv x (in_channels + code_size)]

        output_dict.update({'completion_xyz': completion, 'full_rec': full_rec, 'part_rec': part_rec})
        return output_dict
