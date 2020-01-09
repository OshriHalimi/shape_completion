import torch
from dataset.transforms import Compose,Transform,batch_euclidean_dist_matrix, batch_vnrmls

class Loss(Compose):
    def __init__(self,hparams):
        loss_transforms =[]

        super().__init__(loss_transforms)




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
