import torch
import numpy as np
import pyvista as pv
import math
import cfg
from util.mesh_compute import face_barycenters, mask_indicator
from multiprocessing import Process, Manager
from copy import deepcopy


# ----------------------------------------------------------------------------------------------------------------------#
#                                                    Visualization Functions
# ----------------------------------------------------------------------------------------------------------------------#
def plot_mesh(*args, **kwargs):
    """
    :param v: tensor - A numpy or torch [nv x 3] vertex tensor
    :param f: tensor |  None - (optional) A numpy or torch [nf x 3] vertex tensor OR None
    :param n: tensor |  None - (optional) A numpy or torch [nf x 3] or [nv x3] vertex or face normals. Must input f
    when inputting a face-normal tensor
    :param spheres_on: bool - Plots cloud points as spheres. When f is inputted, does nothing. Default is True
    :param grid_on: bool - Plots an xyz grid with the mesh. Default is False
    :param clr: str or [R,G,B] float list or tensor - Plots  mesh with color clr. clr = v is cool
    :param normal_clr: str or [R,G,B] float list or tensor - Plots  mesh normals with color normal_clr
    :param label: str - (optional) - When inputted, displays a legend with the title label
    :param smooth_shade_on: bool - Plot a smooth version of the facets - just like 3D-Viewer
    :param show_edges: bool - Show edges in black. Only applicable for the full mesh plot
    For color list, see pyvista.plotting.colors
    * For windows keyboard options, see: https://docs.pyvista.org/plotting/plotting.html
    """
    pv.set_plot_theme("document")  # White background
    p = pv.Plotter()
    _append_mesh(p, *args, **kwargs)
    p.show()


def plot_mesh_montage(vb, fb=None, nb=None, labelb=None, spheres_on=True, grid_on=False, clr='lightcoral',
                      normal_clr='lightblue', smooth_shade_on=False, show_edges=False):
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
        _append_mesh(p, v=vb[i], f=f, n=n, label=label, spheres_on=spheres_on, grid_on=grid_on,
                     clr=clr, normal_clr=normal_clr, smooth_shade_on=smooth_shade_on, show_edges=show_edges)

    p.link_views()
    p.show()


# ----------------------------------------------------------------------------------------------------------------------#
#                                                    Helper Functions
# ----------------------------------------------------------------------------------------------------------------------#

def _append_mesh(p, v, f=None, n=None, spheres_on=True, grid_on=False, clr='lightcoral',
                 normal_clr='lightblue', label=None, smooth_shade_on=False, show_edges=False, cmap=None):
    # Align arrays:
    v = v.numpy() if torch.is_tensor(v) else v
    f = f.numpy() if torch.is_tensor(f) else f
    n = n.numpy() if torch.is_tensor(n) else n
    clr = clr.numpy() if torch.is_tensor(clr) else clr
    normal_clr = normal_clr.numpy() if torch.is_tensor(normal_clr) else normal_clr

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
        arrows = pnt_cloud.glyph(orient='normals', scale=False, factor=0.03)
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

class ParallelPlotter(Process):
    from cfg import VIS_CMAP, VIS_METHOD, VIS_SHOW_EDGES, VIS_SMOOTH_SHADING, N_MESH_SETS, VIS_SHOW_GRID

    def __init__(self, shared_dict, faces, n_v):
        super().__init__()
        # self.daemon = True
        # TODO - Add in plot config
        self.sd = shared_dict
        self.last_plotted_epoch = -1

        # Book-keeping:
        self.train_d, self.vfacesal_d = None, None
        self.n_v = n_v
        self.f = faces if self.VIS_METHOD == 'mesh' else None

        self.kwargs = {'smooth_shade_on': self.VIS_SMOOTH_SHADING, 'show_edges': self.VIS_SHOW_EDGES,
                       'spheres_on': (self.VIS_METHOD == 'spheres'), 'cmap': self.VIS_CMAP,
                       'grid_on': self.VIS_SHOW_GRID}

    def run(self):
        # Init on consumer side:
        pv.set_plot_theme("document")
        while 1:
            try:
                if self.sd['poison']:
                    print(f'Pipe poison detected. Exiting plotting supervisor')
                    break
            except (BrokenPipeError, EOFError):  # Missing parent
                print(f'Producer missing. Exiting plotting supervisor')
                break

            current_epoch = self.sd['epoch']
            # if current_epoch == self.last_plotted_epoch:
            #     print(f'\nWarning: geometry for epoch {current_epoch} was already plotted')
            if current_epoch != self.last_plotted_epoch:
                # print(f'\nPlotting geometry for epoch {current_epoch}')
                self.last_plotted_epoch = current_epoch
                self.read_data()
            self.plot_data()

    # Meant to be called by the consumer
    def read_data(self):
        self.train_d, self.val_d = deepcopy(self.sd['data'])

    # Meant to be called by the consumer
    def plot_data(self):

        p = pv.Plotter(shape=(2 * self.N_MESH_SETS, 4), title=f'Visualization for Epoch {self.last_plotted_epoch}')
        for di, (d, set_name) in enumerate(zip([self.train_d, self.val_d], ['Train', 'Vald'])):
            for i in range(self.N_MESH_SETS):
                subplt_row_id = i + di * self.N_MESH_SETS
                mask_ind = mask_indicator(self.n_v, d['gt_mask_vi'][i])
                gtrb = d['gtrb'][i].squeeze()
                gt = d['gt'][i].squeeze()
                tp = d['tp'][i].squeeze()

                # TODO - Add support for normals & P2P
                p.subplot(subplt_row_id, 0)  # GT Reconstructed with colored mask
                _append_mesh(p, v=gtrb, f=self.f, clr=mask_ind, label=f'{set_name} Reconstruction {i}', **self.kwargs)
                p.subplot(subplt_row_id, 1)  # GT with colored mask
                _append_mesh(p, v=gt, f=self.f, clr=mask_ind, label=f'{set_name} GT {i}', **self.kwargs)
                p.subplot(subplt_row_id, 2)  # TP with colored mask
                _append_mesh(p, v=tp, f=self.f, clr=mask_ind, label=f'{set_name} TP {i}', **self.kwargs)
                p.subplot(subplt_row_id, 3)  # GT Reconstructed + Part
                _append_mesh(p, v=gtrb, f=self.f, clr=mask_ind, **self.kwargs)
                # TODO - Remove hard coded 'r'. Do we want to enable part as mesh?
                _append_mesh(p, v=gt[mask_ind, :], f=None, clr='r', label=f'{set_name} Part + Recon {i}', **self.kwargs)

        # p.link_views()
        p.show()

    @staticmethod
    def initialize(faces, n_verts):
        shared_dict = Manager().dict()
        shared_dict['epoch'] = -1
        shared_dict['poison'] = False
        p = ParallelPlotter(shared_dict, faces, n_verts)
        return p

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

    # Meant to be called by the producer
    def finalize(self):
        self.sd['poison'] = True
        print('Workload completed - Please exit plotter to complete execution')
        self.join()


# ----------------------------------------------------------------------------------------------------------------------#
#                                                    Test Suite
# ----------------------------------------------------------------------------------------------------------------------#

def vis_tester():
    from dataset.datasets import PointDatasetMenu, InCfg
    from util.mesh_compute import vnrmls, fnrmls
    ds = PointDatasetMenu.get('FaustPyProj', in_channels=6, in_cfg=InCfg.FULL2PART)
    samp = ds.sample(15)  # dim:
    vv = samp['gt'][0, :, :3]
    ff = ds.faces()
    plot_mesh(v=vv, f=ff, n=vnrmls(vv, ff), label='Badass', grid_on=True)
    # # plot_mesh(v=vv, spheres_on=True, clr=vv)
    # plot_mesh_montage(samp['gt'][:, :, :3], ff)
    # plot_mesh_montage(samp['gt'][:, :, :3])


if __name__ == '__main__':
    from time import sleep

    p = ParallelPlotter.initialize()

    for i in range(1, 2):
        p.push(new_data=i, new_epoch=i)
        sleep(2)

    # p.finalize()
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   GRAVEYARD - Matplotlib
# ----------------------------------------------------------------------------------------------------------------------#
# if self.hparams.mesh_frequency is not None:
#     if self.global_step % self.hparams.mesh_frequency == 0:
#         self.logger.experiment.add_mesh(f"mesh_{self.global_step}", vertices=b['tp'][0, :, :].unsqueeze(0))
# def show_vnormals_matplot(v, f, n):
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, linewidth=1, antialiased=True)
#     ax.quiver(v[:, 0], v[:, 1], v[:, 2], n[:, 0], n[:, 1], n[:, 2], length=0.03, normalize=True)
#     ax.set_aspect('equal', 'box')
#     plt.show()
#
#
# def show_fnormals_matplot(v, f, n):
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
#
#     c = face_barycenters(v, f)
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, linewidth=1, antialiased=True)
#     ax.quiver(c[:, 0], c[:, 1], c[:, 2], n[:, 0], n[:, 1], n[:, 2], length=0.03, normalize=True)
#     ax.set_aspect('equal', 'box')
#     plt.show()
#
# # ----------------------------------------------------------------------------------------------------------------------#
# #                                                   VTK Platform
# # ----------------------------------------------------------------------------------------------------------------------#
# from vtkplotter.actors import Actor
# from vtkplotter.utils import buildPolyData
# def numpy2vtkactor(v, f, clr='gold'):
#     return Actor(buildPolyData(v, f, ), computeNormals=False, c=clr)  # Normals are in C++ - Can't extract them
#
#
# def print_vtkplotter_help():
#     print("""
# ==========================================================
# | Press: i     print info about selected object            |
# |        m     minimise opacity of selected mesh           |
# |        .,    reduce/increase opacity                     |
# |        /     maximize opacity                            |
# |        w/s   toggle wireframe/solid style                |
# |        p/P   change point size of vertices               |
# |        l     toggle edges line visibility                |
# |        x     toggle mesh visibility                      |
# |        X     invoke a cutter widget tool                 |
# |        1-3   change mesh color                           |
# |        4     use scalars as colors, if present           |
# |        5     change background color                     |
# |        0-9   (on keypad) change axes style               |
# |        k     cycle available lighting styles             |
# |        K     cycle available shading styles              |
# |        o/O   add/remove light to scene and rotate it     |
# |        n     show surface mesh normals                   |
# |        a     toggle interaction to Actor Mode            |
# |        j     toggle interaction to Joystick Mode         |
# |        r     reset camera position                       |
# |        C     print current camera info                   |
# |        S     save a screenshot                           |
# |        E     export rendering window to numpy file       |
# |        q     return control to python script             |
# |        Esc   close the rendering window and continue     |
# |        F1    abort execution and exit python kernel      |
# | Mouse: Left-click    rotate scene / pick actors          |
# |        Middle-click  pan scene                           |
# |        Right-click   zoom scene in or out                |
# |        Cntrl-click   rotate scene perpendicularly        |
# |----------------------------------------------------------|
# | Check out documentation at:  https://vtkplotter.embl.es  |
#  ==========================================================""")

#     def show_sample(self, n_shapes=8, key='gt_part', strategy='spheres'):
#
#         using_full = key in ['gt', 'tp']
#         # TODO - Remove this by finding the vtk bug - or replacing the whole vtk shit
#         assert not (not using_full and strategy == 'mesh'), "Mesh strategy for 'part' gets stuck in vtkplotter"
#         fp_fun = self._hi2full_path if using_full else self._hi2proj_path
#
#         samp = self.sample(num_samples=n_shapes, transforms=None)
#         vp = Plotter(N=n_shapes, axes=0)  # ,size="full"
#         vp.legendSize = 0.4
#         for i in range(n_shapes):  # for each available color map name
#
#             name = key.split('_')[0]
#             if using_full:
#                 v, f = samp[key][i, :, 0:3].numpy(), self._f  # TODO - Add in support for faces loaded from file
#             else:
#                 v, f = trunc_to_vertex_subset(samp[name][i, :, 0:3].numpy(), self._f,
#                                               samp[f'{name}_mask_vi'][i])
#
#             if strategy == 'cloud':
#                 a = numpy2vtkactor(v, None, clr='w')  # clr=v is cool
#             elif strategy == 'mesh':
#                 a = numpy2vtkactor(v, f, clr='gold')
#             elif strategy == 'spheres':
#                 a = Spheres(v, c='w', r=0.01)  # TODO - compute r with respect to the mesh
#             else:
#                 raise NotImplementedError
#
#             a.legend(f'{key} | {fp_fun(samp[f"{name}_hi"][i]).name}')
#             vp.show(a, at=i)
#
#         print_vtkplotter_help()
#         vp.show(interactive=1)
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Wisdom Platform
# ----------------------------------------------------------------------------------------------------------------------#
#       if opt.use_visdom:
#         vis = visdom.Visdom(port=8888, env=opt.save_path)
#             # VIZUALIZE
#             if opt.use_visdom and i % 100 == 0:
#                 vis.scatter(X=part[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Train_Part',
#                             opts=dict(title="Train_Part", markersize=2, ), )
#                 vis.scatter(X=template[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Train_Template',
#                             opts=dict(title="Train_Template", markersize=2, ), )
#                 vis.scatter(X=gt_rec[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Train_output',
#                             opts=dict(title="Train_output", markersize=2, ), )
#                 vis.scatter(X=gt[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Train_Ground_Truth',
#                             opts=dict(title="Train_Ground_Truth", markersize=2, ), )
#             vis.line(X=np.column_stack((np.arange(len(Loss_curve_train)), np.arange(len(Loss_curve_val)))),
#                      Y=np.column_stack((np.array(Loss_curve_train), np.array(Loss_curve_val))),
#                      win='Faust loss',
#                      opts=dict(title="Faust loss", legend=["Train loss", "Faust Validation loss", ]))
#             vis.line(X=np.column_stack((np.arange(len(Loss_curve_train)), np.arange(len(Loss_curve_val)))),
#                      Y=np.log(np.column_stack((np.array(Loss_curve_train), np.array(Loss_curve_val)))),
#                      win='"Faust log loss',
#                      opts=dict(title="Faust log loss", legend=["Train loss", "Faust Validation loss", ]))
#
#             vis.line(X=np.column_stack((np.arange(len(Loss_curve_train)), np.arange(len(Loss_curve_val_amass)))),
#                      Y=np.column_stack((np.array(Loss_curve_train), np.array(Loss_curve_val_amass))),
#                      win='AMASS loss',
#                      opts=dict(title="AMASS loss", legend=["Train loss", "Validation loss", "Validation loss amass"]))
#             vis.line(X=np.column_stack((np.arange(len(Loss_curve_train)), np.arange(len(Loss_curve_val_amass)))),
#                      Y=np.log(np.column_stack((np.array(Loss_curve_train), np.array(Loss_curve_val_amass)))),
#                      win='AMASS log loss',
#                      opts=dict(title="AMASS log loss", legend=["Train loss", "Faust Validation loss", ]))
#
