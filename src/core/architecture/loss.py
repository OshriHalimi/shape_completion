import torch
from dataset.transforms import batch_euclidean_dist_matrix, batch_vnrmls
import numpy as np

# p.add_argument('--l2_lambda', nargs=4, type=float, default=[1, 0.1, 0, 0],
#                help='[XYZ,Normal,Moments,Euclid_Maps] L2 loss multiplication modifiers')
# # Loss Modifiers: # TODO - Implement for Euclid Maps as well.
# p.add_argument('--l2_mask_penalty', nargs=3, type=float, default=[0, 0, 0],
#                help='[XYZ,Normal,Moments] increased weight on mask vertices. Use val <= 1 to disable')
# p.add_argument('--l2_distant_v_penalty

# ----------------------------------------------------------------------------------------------------------------------
#                                                   Loss Helpers
# ----------------------------------------------------------------------------------------------------------------------
class Loss:
    # def __init__(self,hparams):
    # 
    #
    #
    #
    # def f2p_loss(xb,gtr):
    #
    #     # Step 1: Align gtr input channels:
    #


    @staticmethod
    def _l2_loss(v1b, v2b, weight_factor, vertex_mask=1):
        return weight_factor * torch.mean(vertex_mask * ((v1b - v2b) ** 2))

    @staticmethod
    def _mask_penalty_weight(mask_b, nv, weight_factor):
        """
        :param mask_b: A list of masks (protected inside a list)
        :param nv: The number of vertices
        :param weight_factor: Additional weight multiplier for the mask vertices - A scalar > 1
        """
        if weight_factor <= 1:
            return 1
        b = len(mask_b)
        w = np.ones((b, nv, 1))
        for i in range(b):
            w[i, mask_b[i][0], :] = weight_factor

    @staticmethod
    def _distant_vertex_weight(gtb_xyz, tpb_xyz, weight_factor):
        """
        :param gtb_xyz: ground-truth batched tensor [b x nv x 3]
        :param tpb_xyz: template batched tensor [b x nv x 3]
        :param weight_factor: Additional weight multiplier for the far off vertices - A scalar > 1
        This function returns a bxnvx1 point-wise weight. For vertices that are similar between gt & tp - return 1.
        For "distant" vertices - return some cost greater than 1.
        Defines the point-wise difference as: d = ||gtb_xyz - tpb_xyz|| - a  [b x nv x 1] vector
        Normalize d by its mean: dhat = d/mean(d)
        Far-off vertices are defined as vertices for which dhat > 1 - i.e., the difference is greater than the mean vertices
        The weight function w is defined by W_i = max(dhat_i,1) * weight_factor
        """
        if weight_factor <= 1:
            return 1
        d = torch.norm(gtb_xyz - tpb_xyz, dim=2, keepdim=True)
        d /= torch.mean(d, dim=1, keepdim=True)  # dhat
        w = torch.max(d, torch.ones((1, 1, 1), device='cuda'))  # TODO - fix device
        w[w > 1] *= weight_factor
        return w




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
            (batch_euclidean_dist_matrix(gt_rec_xyz) - batch_euclidean_dist_matrix(gt_xyz)) ** 2)
        # print(f'Euclid Distances Loss {euclid_dist_loss:4f}')
        loss += euclid_dist_loss

    return loss
