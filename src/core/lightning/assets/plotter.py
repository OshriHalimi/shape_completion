import pyvista as pv
from util.mesh.ops import vertex_mask_indicator
from util.mesh.plots import mesh_append
from multiprocessing import Process, Manager
from copy import deepcopy
from abc import ABC


# ----------------------------------------------------------------------------------------------------------------------#
#                                               Parallel Plot suite
# ----------------------------------------------------------------------------------------------------------------------#

class ParallelPlotterBase(Process, ABC):
    from cfg import VIS_CMAP, VIS_STRATEGY, VIS_SHOW_EDGES, VIS_SMOOTH_SHADING, \
        VIS_N_MESH_SETS, VIS_SHOW_GRID, VIS_SHOW_NORMALS

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

        if old_epoch == -1:  # First push
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


# ----------------------------------------------------------------------------------------------------------------------#
#                                               Parallel Plot suite
# ----------------------------------------------------------------------------------------------------------------------#

class CompletionPlotter(ParallelPlotterBase):
    def prepare_plotter_dict(self, b, network_output):
        # TODO - Generalize this
        gtrb = network_output['completion_xyz']
        max_b_idx = self.VIS_N_MESH_SETS
        dict = {'gt': b['gt'].detach().cpu().numpy()[:max_b_idx, :, :3],
                'tp': b['tp'].detach().cpu().numpy()[:max_b_idx, :, :3],
                'gtrb': gtrb.detach().cpu().numpy()[:max_b_idx],
                'gt_hi': b['gt_hi'][:max_b_idx],
                'tp_hi': b['tp_hi'][:max_b_idx],
                'gt_mask': b['gt_mask'][:max_b_idx]}
        if self.VIS_SHOW_NORMALS:
            dict['gtr_vnb'] = network_output['completion_vnb'].detach().cpu().numpy()[:max_b_idx, :, :]
            dict['gt_vnb'] = b['gt'].detach().cpu().numpy()[:max_b_idx, :, 3:6]
            # dict['gtrb_vnb_is_valid'] = network_output['completion_vnb'].detach().cpu().numpy()[:max_b_idx, :,:]
        return dict

    def plot_data(self):
        # TODO - Generalize this
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
                mesh_append(p, v=gtrb, f=self.f, n=gtr_vnb,
                            clr=mask_ind, label=f'{set_name} Reconstruction {i}', **self.kwargs)
                p.subplot(subplt_row_id, 1)  # GT with colored mask
                mesh_append(p, v=gt, f=self.f, n=gt_vnb,
                            clr=mask_ind, label=f'{set_name} GT {i}', **self.kwargs)
                p.subplot(subplt_row_id, 2)  # TP with colored mask
                mesh_append(p, v=tp, f=self.f, clr=mask_ind, label=f'{set_name} TP {i}', **self.kwargs)
                p.subplot(subplt_row_id, 3)  # GT Reconstructed + Part
                mesh_append(p, v=gtrb, f=self.f, clr=mask_ind, **self.kwargs)
                # TODO - Remove hard coded 'r'. Do we want to enable part as mesh?
                mesh_append(p, v=gt[mask_ind, :], f=None, clr='r', label=f'{set_name} Part + Recon {i}', **self.kwargs)

        # p.link_views()
        p.show()
