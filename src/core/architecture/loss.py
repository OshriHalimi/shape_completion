import torch
from mesh.ops import batch_euclid_dist_mat, batch_vnrmls, batch_vnrmls_, batch_moments
from util.string_op import warn
from mesh.ops import vf_adjacency
from util.torch_nn import PytorchNet
# ----------------------------------------------------------------------------------------------------------------------
#                                                   Loss Helpers
# ----------------------------------------------------------------------------------------------------------------------
class F2PSMPLLoss:
    def __init__(self, hp, f):
        # Copy over from the hyper-params - Remove ties to the hp container for our own editing
        self.lambdas = list(hp.lambdas)
        self.mask_penalties = list(hp.mask_penalties)
        self.dist_v_penalties = list(hp.dist_v_penalties)

        # For CUDA:
        self.dev = hp.dev  # TODO - Might be problematic for multiple GPUs
        self.non_blocking = hp.NON_BLOCKING
        self.def_prec = getattr(torch, hp.UNIVERSAL_PRECISION)

        # Handle Faces:
        if f is not None and self.lambdas[1] > 0:
            self.torch_f = torch.from_numpy(f).long().to(device=self.dev, non_blocking=self.non_blocking)

        # Sanity Check - Input Channels:
        if self.lambdas[1] > 0:
            assert hp.in_channels >= 6, "Only makes sense to compute normal losses with normals available"
        if self.lambdas[2] > 0:
            assert hp.in_channels >= 12, "Only makes sense to compute moment losses with moments available"

        # Sanity Check - Destroy dangling mask_penalties/distant_v_penalties
        for i, lamb in enumerate(self.lambdas[0:3]):  # TODO - Implement 0:5
            if lamb <= 0:
                self.mask_penalties[i] = 0
                self.dist_v_penalties[i] = 0

        # Sanity Check - Check validity of the penalties:
        for i in range(len(self.dist_v_penalties)):
            if 0 < self.dist_v_penalties[i] < 1:
                warn(f'Found an invalid penalty in the distant vertex arg set: at {i} with val '
                     f'{self.dist_v_penalties[i]}.\nPlease use 0 or 1 to remove this specific loss compute')
            if 0 < self.mask_penalties[i] < 1:
                warn(f'Found an invalid penalty in the distant vertex arg set: at {i} with val '
                     f'{self.mask_penalties[i]}.\nPlease use 0 or 1 to remove this specific loss compute')

        # Micro-Optimization - Reduce movement to the GPU:
        if [p for p in self.dist_v_penalties if p > 1]:  # if using_distant_vertex
            self.dist_v_ones = torch.ones((1, 1, 1), device=self.dev, dtype=self.def_prec)

    def compute(self, b, gtrb):
        """
        :param b: The input batch dictionary
        :param gtrb: The batched ground truth reconstruction of dim: [b x nv x 3]
        :return: The loss
        """
        # TODO: Insert support for other out_channels: This codes assumes gtr has 3 input channels
        # Aliasing. We can only assume that channels 0:3 definitely exist
        gtb_xyz = b['gt'][:, :, 0:3]
        tpb_xyz = b['tp'][:, :, 0:3]
        mask_vi = b['gt_mask_vi']
        nv = gtrb.shape[1]

        loss = torch.zeros(1, device=self.dev, dtype=self.def_prec)
        for i, lamb in enumerate(self.lambdas):
            if lamb > 0:
                w = self._mask_penalty_weight(mask_b=mask_vi, nv=nv, p=self.mask_penalties[i]) * \
                    self._distant_vertex_weight(gtb_xyz, tpb_xyz, self.dist_v_penalties[i])
                if i == 0:  # XYZ
                    loss += self._l2_loss(gtb_xyz, gtrb, lamb=lamb, vertex_mask=w)
                elif i == 1:  # Normals
                    vnb, is_valid_vnb = batch_vnrmls(gtrb, self.torch_f)
                    loss += self._l2_loss(b['gt'][:, :, 3:6], vnb, lamb=lamb, vertex_mask=w*is_valid_vnb.unsqueeze(2))
                elif i == 2:  # Moments:
                    loss += self._l2_loss(b['gt'][:, :, 6:12], batch_moments(gtrb), lamb=lamb, vertex_mask=w)
                elif i == 3:  # Euclidean Distance Matrices
                    loss += self._l2_loss(batch_euclid_dist_mat(gtb_xyz), batch_euclid_dist_mat(gtrb), lamb=lamb)
                elif i == 4:  # Face Areas:
                    pass  # TODO - add Face Areas here
                    # loss += self._l2_loss(face_areas(gtb_xyz))
                else:  # TODO - What about volumetric error?
                    raise AssertionError
        return loss

    def _mask_penalty_weight(self, mask_b, nv, p):
        """ TODO - This function was never checked
        :param mask_b: A list of masks (protected inside a list)
        :param nv: The number of vertices
        :param p: Additional weight multiplier for the mask vertices - A scalar > 1
        """
        if p <= 1:
            return 1
        b = len(mask_b)
        w = torch.ones((b, nv, 1), dtype=self.def_prec)
        for i in range(b):
            w[i, mask_b[i][0], :] = p
        return w.to(device=self.dev, non_blocking=self.non_blocking)  # Transfer after looping

    def _distant_vertex_weight(self, gtb_xyz, tpb_xyz, p):
        """ TODO - This function was never checked
        :param gtb_xyz: ground-truth batched tensor [b x nv x 3]
        :param tpb_xyz: template batched tensor [b x nv x 3]
        :param p: Additional weight multiplier for the far off vertices - A scalar > 1
        This function returns a bxnvx1 point-wise weight. For vertices that are similar between gt & tp - return 1.
        For "distant" vertices - return some cost greater than 1.
        Defines the point-wise difference as: d = ||gtb_xyz - tpb_xyz|| - a  [b x nv x 1] vector
        Normalize d by its mean: dhat = d/mean(d)
        Far-off vertices are defined as vertices for which dhat > 1 - i.e.,
        the difference is greater than the mean vertices
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
