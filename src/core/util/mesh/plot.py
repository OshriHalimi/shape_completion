import torch
import numpy as np
import pyvista as pv
import math
from util.mesh.ops import face_barycenters, vertex_mask_indicator
from multiprocessing import Process, Manager
from copy import deepcopy
from abc import ABC


# ----------------------------------------------------------------------------------------------------------------------#
#                                                    Visualization Functions
# ----------------------------------------------------------------------------------------------------------------------#
# noinspection PyIncorrectDocstring
def plot_mesh(v, f=None, n=None, strategy='mesh', grid_on=False, clr='lightcoral', normal_clr='lightblue',
              label=None, smooth_shade_on=False, show_edges=False, cmap=None, normal_scale=1):
    """
    :param v: tensor - A numpy or torch [nv x 3] vertex tensor
    :param f: tensor |  None - (optional) A numpy or torch [nf x 3] vertex tensor OR None
    :param n: tensor |  None - (optional) A numpy or torch [nf x 3] or [nv x3] vertex or face normals. Must input f
    when inputting a face-normal tensor
    :param strategy: One of ['spheres','cloud','mesh']
    :param grid_on: bool - Plots an xyz grid with the mesh. Default is False
    :param clr: str or [R,G,B] float list or tensor - Plots  mesh with color clr. clr = v is cool
    :param normal_clr: str or [R,G,B] float list or tensor - Plots  mesh normals with color normal_clr
    :param label: str - (optional) - When inputted, displays a legend with the title label
    :param smooth_shade_on: bool - Plot a smooth version of the facets - just like 3D-Viewer
    :param show_edges: bool - Show edges in black. Only applicable for strategy == 'mesh'
    For color list, see pyvista.plotting.colors
    * For windows keyboard options, see: https://docs.pyvista.org/plotting/plotting.html
    """
    pv.set_plot_theme("document")  # White background
    p = pv.Plotter()
    _append_mesh(p, v=v, f=f, n=n, grid_on=grid_on, strategy=strategy, clr=clr, normal_clr=normal_clr,
                 label=label, smooth_shade_on=smooth_shade_on, show_edges=show_edges, cmap=cmap,
                 normal_scale=normal_scale)
    p.show()


# noinspection PyIncorrectDocstring
def plot_mesh_montage(vb, fb=None, nb=None, strategy='mesh', labelb=None, grid_on=False, clr='lightcoral',
                      normal_clr='lightblue', smooth_shade_on=False, show_edges=False, normal_scale=1):
    """
    :param vb: tensor | list - [b x nv x 3] batch of meshes or list of length b with tensors [nvx3]
    :param fb: tensor | list | None - (optional) [b x nf x 3]
    batch of face indices OR a list of length b with tensors [nfx3]
    OR a [nf x 3] in the case of a uniform face array for all meshes
    :param nb: tensor | list | None - (optional) [b x nf|nv x 3]  batch of normals. See above
    :param labelb: list of titles for each mesh, or None
    * For other arguments, see plot_mesh
    * For windows keyboard options, see: https://docs.pyvista.org/plotting/plotting.html
    """
    n_meshes = vb.shape[0]
    pv.set_plot_theme("document")  # White background
    n_rows = math.floor(math.sqrt(n_meshes))
    n_cols = math.ceil(n_meshes / n_rows)

    shape = (n_rows, n_cols)
    p = pv.Plotter(shape=shape)
    r, c = np.unravel_index(range(n_meshes), shape)

    for i in range(n_meshes):
        f = fb if fb is None or (hasattr(fb, 'shape') and len(fb.shape) == 2) else fb[i]
        # Uniform faces support. fb[i] is equiv to fb[i,:,:]
        n = nb if nb is None else nb[i]
        label = labelb if labelb is None else labelb[i]
        p.subplot(r[i], c[i])
        _append_mesh(p, v=vb[i], f=f, n=n, strategy=strategy, label=label, grid_on=grid_on, normal_scale=normal_scale,
                     clr=clr, normal_clr=normal_clr, smooth_shade_on=smooth_shade_on, show_edges=show_edges)

    p.link_views()
    p.show()


# ----------------------------------------------------------------------------------------------------------------------#
#                                                    Helper Functions
# ----------------------------------------------------------------------------------------------------------------------#

def _append_mesh(p, v, f=None, n=None, strategy='mesh', grid_on=False, clr='lightcoral',
                 normal_clr='lightblue', label=None, smooth_shade_on=False, show_edges=False, cmap=None,
                 normal_scale=1):
    # Align arrays:
    v = v.numpy() if torch.is_tensor(v) else v
    f = f.numpy() if torch.is_tensor(f) else f
    n = n.numpy() if torch.is_tensor(n) else n
    clr = clr.numpy() if torch.is_tensor(clr) else clr
    normal_clr = normal_clr.numpy() if torch.is_tensor(normal_clr) else normal_clr

    # Align strategy
    if strategy == 'mesh':
        assert f is not None, "Must supply faces for mesh strategy"
    else:
        f = None  # Destroy the face information
    spheres_on = (strategy == 'spheres')

    # Create Data object:
    if f is not None:
        # Adjust f to the needed format
        pnt_cloud = pv.PolyData(v, np.concatenate((np.full((f.shape[0], 1), 3), f), 1))
    else:
        pnt_cloud = pv.PolyData(v)

    # Default size for spheres & pnt clouds
    point_size = 6.0 if spheres_on else 2.0  # TODO - Dynamic computation of this, based on mesh volume

    # Handle difference between color and scalars, to support RGB tensor
    if isinstance(clr, str) or len(clr) == 3:
        color = clr
        scalars = None
        clr_str = clr
    else:
        color = None
        clr_str = 'w'
        scalars = clr

        # Add the meshes to the plotter:
    p.add_mesh(pnt_cloud, smooth_shading=smooth_shade_on, scalars=scalars, color=color, cmap=cmap,
               show_edges=show_edges,  # For full mesh visuals - ignored on point cloud plots
               render_points_as_spheres=spheres_on, point_size=point_size)  # For sphere visuals - ignored on full mesh

    p.camera_position = [(0, 0, 4.5), (0, 0, 0), (0, 1, 0)]
    if n is not None:  # Face normals or vertex normals
        if not n.shape[0] == v.shape[0]:  # Face normals
            assert f is not None and n.shape[0] == f.shape[0]  # Faces are required for compute
            pnt_cloud = pv.PolyData(face_barycenters(v, f))
        pnt_cloud['normals'] = n
        # noinspection PyTypeChecker
        arrows = pnt_cloud.glyph(orient='normals', scale=False, factor=0.03 * normal_scale)
        # TODO - Dynamic computation of normal_scale
        if isinstance(normal_clr, str) or len(normal_clr) == 3:
            color = normal_clr
            scalars = None
        else:
            color = None
            scalars = normal_clr
        p.add_mesh(arrows, color=color, scalars=scalars)

    # Book-keeping:
    if label is not None and label:
        siz = 0.25
        p.add_legend(labels=[(label, clr_str)], size=[siz, siz / 3])
    if grid_on:
        p.show_grid()


# ----------------------------------------------------------------------------------------------------------------------#
#                                               Parallel Plot suite
# ----------------------------------------------------------------------------------------------------------------------#

class ParallelPlotterBase(Process, ABC):
    from cfg import VIS_CMAP, VIS_STRATEGY, VIS_SHOW_EDGES, VIS_SMOOTH_SHADING, \
        VIS_N_MESH_SETS, VIS_SHOW_GRID,VIS_SHOW_NORMALS

    def __init__(self, faces, n_verts):
        super().__init__()
        self.sd = Manager().dict()
        self.sd['epoch'] = -1
        self.sd['poison'] = False

        self.n_v = n_verts
        self.last_plotted_epoch = -1
        self.f = faces if self.VIS_STRATEGY == 'mesh' else None
        self.train_d, self.val_d, self.data_cache, self.plt_title = None, None, None, None

        self.kwargs = {'smooth_shade_on': self.VIS_SMOOTH_SHADING, 'show_edges': self.VIS_SHOW_EDGES,
                       'strategy': self.VIS_STRATEGY, 'cmap': self.VIS_CMAP,
                       'grid_on': self.VIS_SHOW_GRID}

    def run(self):
        # Init on consumer side:
        pv.set_plot_theme("document")
        while 1:
            try:
                if self.last_plotted_epoch != -1 and self.sd['poison']:  # Plotted at least one time + Poison
                    print(f'Pipe poison detected. Displaying one last time, and then exiting plotting supervisor')
                    self.try_update_data(final=True)
                    self.plot_data()
                    break
            except (BrokenPipeError, EOFError):  # Missing parent
                print(f'Producer missing. Exiting plotting supervisor')
                break

            self.try_update_data()
            self.plot_data()

    # Meant to be called by the consumer
    def try_update_data(self, final=False):
        current_epoch = self.sd['epoch']
        if current_epoch != self.last_plotted_epoch:
            self.last_plotted_epoch = current_epoch
            self.train_d, self.val_d = deepcopy(self.sd['data'])
        if final:
            self.plt_title = f'Final visualization before closing for Epoch {self.last_plotted_epoch}'
        else:
            self.plt_title = f'Visualization for Epoch {self.last_plotted_epoch}'

            # Update version with one single read
        # Slight problem of atomicity here - with respect to current_epoch. Data may have changed in the meantime -
        # but it is still new data, so no real problem. May be resolved with lock = manager.Lock() before every update->
        # lock.acquire() / lock.release(), but this will be problematic for the main process - which we want to optimize

    # Meant to be called by the consumer
    def plot_data(self):
        raise NotImplementedError

    # Meant to be called by the producer
    def push(self, new_epoch, new_data):
        # new_data = (train_dict,vald_dict)
        old_epoch = self.sd['epoch']
        assert new_epoch != old_epoch

        # Update shared data (possibly before process starts)
        self.sd['data'] = new_data
        self.sd['epoch'] = new_epoch

        if old_epoch == 0:  # First push
            self.start()

    def cache(self, data):
        self.data_cache = data

    def uncache(self):
        cache = self.data_cache
        self.data_cache = None
        return cache
        # Meant to be called by the producer

    def finalize(self):
        self.sd['poison'] = True
        print('Workload completed - Please exit plotter to complete execution')
        self.join()


class CompletionPlotter(ParallelPlotterBase):
    def plot_data(self):
        gtr_vnb = None
        gt_vnb = None
        p = pv.Plotter(shape=(2 * self.VIS_N_MESH_SETS, 4), title=self.plt_title)
        for di, (d, set_name) in enumerate(zip([self.train_d, self.val_d], ['Train', 'Vald'])):
            for i in range(self.VIS_N_MESH_SETS):
                subplt_row_id = i + di * self.VIS_N_MESH_SETS
                mask_ind = vertex_mask_indicator(self.n_v, d['gt_mask'][i])
                gtrb = d['gtrb'][i].squeeze()
                gt = d['gt'][i].squeeze()
                tp = d['tp'][i].squeeze()
                if self.VIS_SHOW_NORMALS:
                    gtr_vnb = d['gtr_vnb'][i].squeeze()
                    gt_vnb = d['gt_vnb'][i].squeeze()

                # TODO - Add support for normals & P2P
                # TODO - Check why in mesh method + tensor colors, colors are interpolated onto the faces.
                p.subplot(subplt_row_id, 0)  # GT Reconstructed with colored mask
                _append_mesh(p, v=gtrb, f=self.f,n=gtr_vnb,
                             clr=mask_ind, label=f'{set_name} Reconstruction {i}', **self.kwargs)
                p.subplot(subplt_row_id, 1)  # GT with colored mask
                _append_mesh(p, v=gt, f=self.f,n=gt_vnb,
                             clr=mask_ind, label=f'{set_name} GT {i}', **self.kwargs)
                p.subplot(subplt_row_id, 2)  # TP with colored mask
                _append_mesh(p, v=tp, f=self.f, clr=mask_ind, label=f'{set_name} TP {i}', **self.kwargs)
                p.subplot(subplt_row_id, 3)  # GT Reconstructed + Part
                _append_mesh(p, v=gtrb, f=self.f, clr=mask_ind, **self.kwargs)
                # TODO - Remove hard coded 'r'. Do we want to enable part as mesh?
                _append_mesh(p, v=gt[mask_ind, :], f=None, clr='r', label=f'{set_name} Part + Recon {i}', **self.kwargs)

        # p.link_views()
        p.show()


# ----------------------------------------------------------------------------------------------------------------------#
#                                                    Test Suite
# ----------------------------------------------------------------------------------------------------------------------#

def visuals_tester():
    from dataset.datasets import FullPartDatasetMenu
    ds = FullPartDatasetMenu.get('FaustPyProj')
    samp = ds.sample(15)  # dim:
    vv = samp['gt'][0, :, :3]
    ff = ds.faces()
    # plot_mesh(v=vv, f=ff, n=fnrmls(vv, ff), label='Badass', grid_on=True)
    # # plot_mesh(v=vv, spheres_on=True, clr=vv)
    # plot_mesh_montage(samp['gt'][:, :, :3], ff)
    plot_mesh_montage(samp['gt'][:, :, :3], strategy='spheres')


if __name__ == '__main__':
    visuals_tester()
