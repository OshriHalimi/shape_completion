import numpy as np
from abc import ABC
import sys
import os
from external_tools.pyrender.lib import render

sys.path.append(os.path.abspath(os.path.join('..', 'core')))  # For core.utils
from util.mesh.plots import plot_mesh_montage, plot_mesh
from util.fs import restart_program


class DeformationFailureError(Exception):
    pass


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#
class Deformation(ABC):
    def deform(self, v, f):
        # Must return list of dictionaries
        raise NotImplementedError

    def name(self):
        return self.__class__.__name__.lower()

    def num_expected_deformations(self):
        # Must return int
        raise NotImplementedError

    def reset(self):
        raise DeformationFailureError  # Presuming that a failure is not fixable

    def needs_validation(self):
        # Must return bool
        raise NotImplementedError


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#
class Projection(Deformation):
    DANGEROUS_MASK_LEN = 100

    def __init__(self, max_angs=10, angs_to_take=None, elevation=None):
        # TODO - Implement elevation compute
        if angs_to_take is None:
            angs_to_take = max_angs
        else:
            assert angs_to_take <= max_angs
        self.angs_to_take = angs_to_take
        self.num_angles = max_angs
        self.range = np.arange(max_angs)
        # Needed by Render
        self.render_info = {'Height': 480, 'Width': 640, 'fx': 575, 'fy': 575, 'cx': 319.5, 'cy': 239.5}
        if os.name!='nt':
            render.setup(self.render_info)
        self.world2cam_mats = self._prep_world2cam_mats()

    def deform(self, v, f):

        assert v.dtype == 'float32' and f.dtype == 'int32'

        masks = []
        if self.angs_to_take == self.num_angles:
            angle_ids = self.range
        else:
            angle_ids = sorted(np.random.choice(self.range, size=self.angs_to_take, replace=False))

        for angi in angle_ids:
            # render.setup(self.render_info)
            context = render.set_mesh(v, f)
            render.render(context, self.world2cam_mats[angi])
            mask, _, _ = render.get_vmap(context, self.render_info)  # vindices, vweights, findices
            mask = np.unique(mask)
            # Sanity:
            if len(mask) < self.DANGEROUS_MASK_LEN:
                masks.append(None)
            else:
                masks.append({'mask': mask, 'angle_id': angi})

            # render.clear()

        return masks

    def name(self,full=True):
        if full:
            return f'{super().name()}_{self.angs_to_take}_of_{self.num_angles}_angs'
        else:
            return super().name()

    def num_expected_deformations(self):
        return self.angs_to_take

    def reset(self):
        restart_program()
        # render.reset()
        # render.setup(self.render_info)

    def needs_validation(self):
        return True

    def _prep_world2cam_mats(self):

        cam2world = np.array([[0.85408425, 0.31617427, -0.375678, 0.56351697 * 2],
                              [0., -0.72227067, -0.60786998, 0.91180497 * 2],
                              [-0.52013469, 0.51917219, -0.61688, 0.92532003 * 2],
                              [0., 0., 0., 1.]], dtype=np.float32)

        # rotate the mesh elevation by 30 degrees
        Rx = np.array([[1, 0, 0, 0],
                       [0., np.cos(np.pi / 6), -np.sin(np.pi / 6), 0],
                       [0, np.sin(np.pi / 6), np.cos(np.pi / 6), 0],
                       [0., 0., 0., 1.]], dtype=np.float32)
        cam2world = np.matmul(Rx, cam2world)

        world2cam_mats = []
        for i_ang, ang in enumerate(np.linspace(0, 2 * np.pi, self.num_angles)):
            Ry = np.array([[np.cos(ang), 0, -np.sin(ang), 0],
                           [0., 1, 0, 0],
                           [np.sin(ang), 0, np.cos(ang), 0],
                           [0., 0., 0., 1.]], dtype=np.float32)
            world2cam = np.linalg.inv(np.matmul(Ry, cam2world)).astype('float32')
            world2cam_mats.append(world2cam)
        return world2cam_mats
