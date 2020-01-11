import torch
from dataset.transforms import batch_euclid_dist_mat, batch_vnrmls, batch_moments
import numpy as np


# p.add_argument('--lambdas', nargs=4, type=float, default=[1, 0.1, 0, 0],
#                help='[XYZ,Normal,Moments,Euclid_Maps] loss multiplication modifiers')
# # Loss Modifiers: # TODO - Implement for Euclid Maps as well.
# p.add_argument('--mask_penalties', nargs=3, type=float, default=[0, 0, 0],
#                help='[XYZ,Normal,Moments] increased weight on mask vertices. Use val <= 1 to disable')
# p.add_argument('--dist_v_penalties', nargs=3, type=float, default=[0, 0, 0],
#                help='[XYZ,Normal,Moments] increased weight on distant vertices. Use val <= 1 to disable')

# ----------------------------------------------------------------------------------------------------------------------
#                                                   Loss Helpers
# ----------------------------------------------------------------------------------------------------------------------
class F2PLoss:
    def __init__(self, hparams, device):

        # Take all deciding factors:
        self.in_channels = hparams.in_channels
        self.lambdas = hparams.lambdas
        self.mask_penalties = hparams.mask_penalties
        self.dist_v_penalties = hparams.dist_v_penalties

        # Sanity Check - Input Channels:
        if self.lambdas[1] > 0:
            assert self.in_channels >= 6, "Only makes sense to compute normal losses with normals available"
        if self.lambdas[2] > 0:
            assert self.in_channels >= 12, "Only makes sense to compute moment losses with moments available"

        # Sanity Check - Destory dangling mask_penalties/distant_v_penalties
        for i, lamb in enumerate(self.lambdas[0:3]):  # TODO - Implement 0:4
            if lamb <= 0:
                self.mask_penalties[i] = 0
                self.dist_v_penalties[i] = 0

        self.device = device  # TODO - Is torch.current enough? Don't think so...

    def compute(self, b, gtrb):

        # TODO - This codes assumes gtr has 3 input channels

        # Aliasing:
        gtb_xyz = b['gt_v'][:, 0:3, :]
        tpb_xyz = b['tp_v'][:, 0:3, :]
        mask_vi = b['gt_mask_vi']
        nv = gtrb.shape[1]

        loss = torch.zeros((1), device=self.device)
        for i, lamb in enumerate(self.lambdas):

            if lamb > 0:
                w = self._mask_penalty_weight(mask_b=mask_vi, nv=nv, lamb=self.mask_penalties[i]) * \
                    self._distant_vertex_weight(gtb_xyz, tpb_xyz, self.dist_v_penalties[i])
                if i == 0:  # XYZ
                    loss += self._l2_loss(gtb_xyz, gtrb, lamb=lamb, vertex_mask=w)
                elif i == 1:  # Normals
                    loss += self._l2_loss(b['gt_v'][:, 3:6, :], batch_vnrmls(gtrb, b['f']), lamb=lamb,
                                          vertex_mask=w)
                elif i == 2:  # Moments:
                    loss += self._l2_loss(b['gt_v'][:, 6:12, :], batch_moments(gtrb), lamb=lamb, vertex_mask=w)
                elif i==3:
                    loss += self._l2_loss(batch_euclid_dist_mat(gtb_xyz),batch_euclid_dist_mat(gtrb),lamb=lamb)
        return loss

    def _mask_penalty_weight(self, mask_b, nv, lamb):
        """
        :param mask_b: A list of masks (protected inside a list)
        :param nv: The number of vertices
        :param lamb: Additional weight multiplier for the mask vertices - A scalar > 1
        """
        if lamb <= 1:
            return 1
        b = len(mask_b)
        w = torch.ones((b, nv, 1), device=self.device)
        for i in range(b):
            w[i, mask_b[i][0], :] = lamb

    def _distant_vertex_weight(self, gtb_xyz, tpb_xyz, lamb):
        """
        :param gtb_xyz: ground-truth batched tensor [b x nv x 3]
        :param tpb_xyz: template batched tensor [b x nv x 3]
        :param lamb: Additional weight multiplier for the far off vertices - A scalar > 1
        This function returns a bxnvx1 point-wise weight. For vertices that are similar between gt & tp - return 1.
        For "distant" vertices - return some cost greater than 1.
        Defines the point-wise difference as: d = ||gtb_xyz - tpb_xyz|| - a  [b x nv x 1] vector
        Normalize d by its mean: dhat = d/mean(d)
        Far-off vertices are defined as vertices for which dhat > 1 - i.e., the difference is greater than the mean vertices
        The weight function w is defined by W_i = max(dhat_i,1) * lamb
        """
        if lamb <= 1:
            return 1
        d = torch.norm(gtb_xyz - tpb_xyz, dim=2, keepdim=True)
        d /= torch.mean(d, dim=1, keepdim=True)  # dhat
        w = torch.max(d, torch.ones((1, 1, 1), device=self.device))  # TODO - fix device
        w[w > 1] *= lamb
        return w

    @staticmethod
    def _l2_loss(v1b, v2b, lamb, vertex_mask=1):
        return lamb * torch.mean(vertex_mask * ((v1b - v2b) ** 2))


# TODO - This needs work
def compute_loss(gt, gt_rec, template, mask_loss, f, opt):
    gt_rec_xyz = gt_rec[:, :3, :]
    gt_xyz = gt[:, :3, :]
    template_xyz = template[:, :3, :]

    # Compute XYZ Loss
    multiplying_factor = 1
    if opt.mask_xyz_penalty and mask_loss is not None:
        multiplying_factor *= mask_loss
    if opt.distant_vertex_loss_slope > 0:
        distant_vertex_penalty = torch.norm(gt_xyz - template_xyz, dim=1, keepdim=True)  # Vector
        distant_vertex_penalty /= torch.mean(distant_vertex_penalty, dim=2, keepdim=True)
        distant_vertex_penalty = torch.max(distant_vertex_penalty, torch.ones((1, 1, 1), device='cuda'))
        distant_vertex_penalty[distant_vertex_penalty > 1] *= opt.distant_vertex_loss_slope
        # print(f'Distant Vertex Loss {distant_vertex_loss:4f}')
        multiplying_factor *= distant_vertex_penalty
    loss = torch.mean(multiplying_factor * ((gt_rec_xyz - gt_xyz) ** 2))

    # Compute Normal Loss
    if opt.normal_loss_slope > 0:
        gt_rec_n = batch_vnrmls(gt_rec_xyz, f)
        if gt.shape[1] > 3:  # Has normals
            gt_n = gt[:, 3:6, :]
        else:
            gt_n = batch_vnrmls(gt_xyz, f)

        multiplying_factor = 1
        if opt.use_mask_normal_penalty and mask_loss is not None:
            multiplying_factor *= mask_loss
        if opt.use_mask_normal_distant_vertex_penalty:
            multiplying_factor *= distant_vertex_penalty

        normal_loss = opt.normal_loss_slope * torch.mean(multiplying_factor * ((gt_rec_n - gt_n) ** 2))
        # print(f'Vertex Normal Loss {normal_loss:4f}')
        loss += normal_loss

    # Compute Euclidean Distance Loss
    if opt.euclid_dist_loss_slope > 0:
        euclid_dist_loss = opt.euclid_dist_loss_slope * torch.mean(
            (batch_euclid_dist_mat(gt_rec_xyz) - batch_euclid_dist_mat(gt_xyz)) ** 2)
        # print(f'Euclid Distances Loss {euclid_dist_loss:4f}')
        loss += euclid_dist_loss

    return loss
