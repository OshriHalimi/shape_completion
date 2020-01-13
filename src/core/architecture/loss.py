import torch
from dataset.transforms import batch_euclid_dist_mat, batch_vnrmls, batch_moments
from util.gen import warn


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Target Arguments
# ----------------------------------------------------------------------------------------------------------------------
# p.add_argument('--lambdas', nargs=4, type=float, default=[1, 0.1, 0, 0],
#                help='[XYZ,Normal,Moments,Euclid_Maps] loss multiplication modifiers')
# # Loss Modifiers:
# p.add_argument('--mask_penalties', nargs=3, type=float, default=[0, 0, 0],
#                help='[XYZ,Normal,Moments] increased weight on mask vertices. Use val <= 1 to disable')
# p.add_argument('--dist_v_penalties', nargs=3, type=float, default=[0, 0, 0],
#                help='[XYZ,Normal,Moments] increased weight on distant vertices. Use val <= 1 to disable')

# ----------------------------------------------------------------------------------------------------------------------
#                                                   Loss Helpers
# ----------------------------------------------------------------------------------------------------------------------
class F2PSMPLLoss:
    def __init__(self, hparams, faces, device):

        # Take all deciding factors:
        self.in_channels = hparams.in_channels
        self.lambdas = hparams.lambdas
        self.mask_penalties = hparams.mask_penalties
        self.dist_v_penalties = hparams.dist_v_penalties
        self.f = faces
        self.device = device

        # Sanity Check - Input Channels:
        if self.lambdas[1] > 0:
            assert self.in_channels >= 6, "Only makes sense to compute normal losses with normals available"
        if self.lambdas[2] > 0:
            assert self.in_channels >= 12, "Only makes sense to compute moment losses with moments available"

        # Sanity Check - Destroy dangling mask_penalties/distant_v_penalties
        for i, lamb in enumerate(self.lambdas[0:3]):  # TODO - Implement 0:4
            if lamb <= 0:
                self.mask_penalties[i] = 0
                self.dist_v_penalties[i] = 0

        # Sanity Check - Check validity of the penalties:
        for i in range(len(self.dist_v_penalties)):
            if 0 < self.dist_v_penalties[i] < 1:
                warn(f'Found an invalid penalty in the distant vertex arg set: at {i} with val '
                     f'{self.dist_v_penalties[i]}.\nPlease use 0 or 1 to shut off the method')
            if 0 < self.mask_penalties[i] < 1:
                warn(f'Found an invalid penalty in the distant vertex arg set: at {i} with val '
                     f'{self.mask_penalties[i]}.\nPlease use 0 or 1 to shut off the method')

        # Micro-Optimization - Reduce movement to the GPU:
        if [p for p in self.dist_v_penalties if p > 1]:  # if using_distant_vertex
            self.dist_v_ones = torch.ones((1, 1, 1), device=self.device)  # TODO

    def compute(self, b, gtrb):
        """
        :param b: The input batch dictionaryz
        :param gtrb: The batched ground truth reconstruction of dim: [b x nv x 3]
        :return:
        """
        # TODO: Insert support for other out_channels: This codes assumes gtr has 3 input channels
        # Aliasing:
        gtb_xyz = b['gt_v'][:, :, 0:3]
        tpb_xyz = b['tp_v'][:, :, 0:3]
        mask_vi = b['gt_mask_vi']
        nv = gtrb.shape[1]

        loss = torch.zeros((1), device=self.device)
        for i, lamb in enumerate(self.lambdas):
            if lamb > 0:
                w = self._mask_penalty_weight(mask_b=mask_vi, nv=nv, p=self.mask_penalties[i]) * \
                    self._distant_vertex_weight(gtb_xyz, tpb_xyz, self.dist_v_penalties[i])
                if i == 0:  # XYZ
                    loss += self._l2_loss(gtb_xyz, gtrb, lamb=lamb, vertex_mask=w)
                elif i == 1:  # Normals
                    loss += self._l2_loss(b['gt_v'][:, :, 3:6], batch_vnrmls(gtrb, self.f), lamb=lamb, vertex_mask=w)
                elif i == 2:  # Moments:
                    loss += self._l2_loss(b['gt_v'][:, :, 6:12], batch_moments(gtrb), lamb=lamb, vertex_mask=w)
                elif i == 3:  # Euclidean Distance Matrices
                    loss += self._l2_loss(batch_euclid_dist_mat(gtb_xyz), batch_euclid_dist_mat(gtrb), lamb=lamb)
                elif i == 4:  # Face Areas:
                    pass
                # TODO - add Face Areas here:
                # loss += self._l2_loss(face_areas(gtb_xyz))
                else:
                    raise AssertionError
        return loss

    def _mask_penalty_weight(self, mask_b, nv, p):
        """
        :param mask_b: A list of masks (protected inside a list)
        :param nv: The number of vertices
        :param p: Additional weight multiplier for the mask vertices - A scalar > 1
        """
        if p <= 1:
            return 1
        b = len(mask_b)
        w = torch.ones((b, nv, 1))
        for i in range(b):
            w[i, mask_b[i][0], :] = p
        return w.cuda(device=self.device)  # TODO

    def _distant_vertex_weight(self, gtb_xyz, tpb_xyz, p):
        """
        :param gtb_xyz: ground-truth batched tensor [b x nv x 3]
        :param tpb_xyz: template batched tensor [b x nv x 3]
        :param p: Additional weight multiplier for the far off vertices - A scalar > 1
        This function returns a bxnvx1 point-wise weight. For vertices that are similar between gt & tp - return 1.
        For "distant" vertices - return some cost greater than 1.
        Defines the point-wise difference as: d = ||gtb_xyz - tpb_xyz|| - a  [b x nv x 1] vector
        Normalize d by its mean: dhat = d/mean(d)
        Far-off vertices are defined as vertices for which dhat > 1 - i.e., the difference is greater than the mean vertices
        The weight function w is defined by W_i = max(dhat_i,1) * lamb
        """
        if p <= 1:
            return 1
        d = torch.norm(gtb_xyz - tpb_xyz, dim=2, keepdim=True)
        d /= torch.mean(d, dim=1, keepdim=True)  # dhat
        w = torch.max(d, self.dist_v_ones)
        w[w > 1] *= p
        return w  # Promised tensor

    @staticmethod
    def _l2_loss(v1b, v2b, lamb, vertex_mask=1):
        return lamb * torch.mean(vertex_mask * ((v1b - v2b) ** 2))

# ----------------------------------------------------------------------------------------------------------------------
#                                                 Graveyard
# ----------------------------------------------------------------------------------------------------------------------
# def compute_loss(gt, gt_rec, template, mask_loss, f, opt):
#     gt_rec_xyz = gt_rec[:, :3, :]
#     gt_xyz = gt[:, :3, :]
#     template_xyz = template[:, :3, :]
#
#     # Compute XYZ Loss
#     multiplying_factor = 1
#     if opt.mask_xyz_penalty and mask_loss is not None:
#         multiplying_factor *= mask_loss
#     if opt.distant_vertex_loss_slope > 0:
#         distant_vertex_penalty = torch.norm(gt_xyz - template_xyz, dim=1, keepdim=True)  # Vector
#         distant_vertex_penalty /= torch.mean(distant_vertex_penalty, dim=2, keepdim=True)
#         distant_vertex_penalty = torch.max(distant_vertex_penalty, torch.ones((1, 1, 1), device='cuda'))
#         distant_vertex_penalty[distant_vertex_penalty > 1] *= opt.distant_vertex_loss_slope
#         # print(f'Distant Vertex Loss {distant_vertex_loss:4f}')
#         multiplying_factor *= distant_vertex_penalty
#     loss = torch.mean(multiplying_factor * ((gt_rec_xyz - gt_xyz) ** 2))
#
#     # Compute Normal Loss
#     if opt.normal_loss_slope > 0:
#         gt_rec_n = batch_vnrmls(gt_rec_xyz, f)
#         if gt.shape[1] > 3:  # Has normals
#             gt_n = gt[:, 3:6, :]
#         else:
#             gt_n = batch_vnrmls(gt_xyz, f)
#
#         multiplying_factor = 1
#         if opt.use_mask_normal_penalty and mask_loss is not None:
#             multiplying_factor *= mask_loss
#         if opt.use_mask_normal_distant_vertex_penalty:
#             multiplying_factor *= distant_vertex_penalty
#
#         normal_loss = opt.normal_loss_slope * torch.mean(multiplying_factor * ((gt_rec_n - gt_n) ** 2))
#         # print(f'Vertex Normal Loss {normal_loss:4f}')
#         loss += normal_loss
#
#     # Compute Euclidean Distance Loss
#     if opt.euclid_dist_loss_slope > 0:
#         euclid_dist_loss = opt.euclid_dist_loss_slope * torch.mean(
#             (batch_euclid_dist_mat(gt_rec_xyz) - batch_euclid_dist_mat(gt_xyz)) ** 2)
#         # print(f'Euclid Distances Loss {euclid_dist_loss:4f}')
#         loss += euclid_dist_loss
#
#     return loss
