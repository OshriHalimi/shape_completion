import os
import sys
import shutil
import numpy as np
import pickle
import json
from tqdm import tqdm
from deformations import Projection
from pathlib import Path
import time
import tempfile
from types import MethodType

# import random

sys.path.append(os.path.abspath(os.path.join('..', 'core')))
from util.strings import banner, print_warning, print_error, title
from util.mesh.ios import read_obj_verts, read_ply_verts
from util.mesh.plots import plot_mesh_montage, plot_mesh
from util.mesh.ops import box_center
from util.fs import assert_new_dir

# ----------------------------------------------------------------------------------------------------------------------#
#                                                       Globals
# ----------------------------------------------------------------------------------------------------------------------#
ROOT = (Path(__file__).parents[0]).resolve()
OUTPUT_ROOT = ROOT / 'deformation_outputs'
COLLATERALS_DIR = ROOT / 'collaterals'


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#

def project_mixamo_main():
    if os.name == 'nt':
        in_dp = Path(r'Z:\ShapeCompletion\Mixamo\Blender\MPI-FAUST')
    else:  # Presuming Linux
        in_dp = Path(r"/usr/samba_mount/ShapeCompletion/Mixamo/Blender/MPI-FAUST")

    banner('MIXAMO Projection')
    deformer = Projection(max_angs=10, angs_to_take=2)
    m = MixamoCreator(deformer, in_dp, shape_frac_from_vgroup=1)
    for sub in m.subjects():
        m.deform_subject(sub=sub)


def project_smal_main():
    banner('SMAL Projection')
    deformer = Projection(max_angs=10, angs_to_take=10)
    m = SMALCreator(deformer, Path(r'C:\Users\idoim\Desktop\ShapeCompletion\data\synthetic\SMALTestPyProj\full'))
    for sub in m.subjects():
        m.deform_subject(sub=sub)


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#
class DataCreator:
    MIN_VGROUP_SUCCESS_FRAC = 0.5  # Under this, the validation group will be considered a failure
    MIN_VGROUP_SIZE = 10  # Under this, the validation group will be considered too small to take
    MAX_FAILED_VGROUPS_BEFORE_RESET = 7  # For PyRender deformation only - maximum amount of failures before reset

    TMP_ROOT = tempfile._get_default_tempdir()
    OUT_ROOT = Path(OUTPUT_ROOT)  # May be overridden
    RECORD_ROOT = OUT_ROOT  # May be overridden

    COLLAT_DP = Path(COLLATERALS_DIR)

    def __init__(self, deformer, in_dp, shape_frac_from_vgroup=1):

        # Sanity:
        self.in_dp = Path(in_dp)
        assert self.OUT_ROOT.is_dir(), f"Could not find directory {self.OUT_ROOT}"
        assert self.in_dp.id_dir(), f"Could not find directory {self.in_dp}"

        # Set deformer:
        self.deformer = deformer
        self.shape_frac_from_vgroup = shape_frac_from_vgroup
        ds_id, deform_id = self.dataset_name(), self.deform_name()
        self.read_shape = MethodType(getattr(self, f'read_shape_for_{deform_id.lower()}'), self)

        # Create dirs for the dataset + specific deformation:
        self.out_dp = self.OUT_ROOT / ds_id / deform_id / 'outputs'
        self.out_dp.mkdir(parents=True, exist_ok=True)
        print(f'Target output directory: {self.out_dp}')

        if deformer.needs_validation():
            assert self.TMP_ROOT.is_dir(), f"Could not find directory {self.TMP_ROOT}"
            self.tmp_dp = self.TMP_ROOT / ds_id / deform_id
            self.tmp_dp.mkdir(parents=True, exist_ok=True)
            self.record_dp = self.RECORD_ROOT / ds_id / deform_id / 'record'
            self.record_dp.mkdir(parents=True, exist_ok=True)
            print(f'Target validation-records directory: {self.record_dp}')

    def dataset_name(self):
        return self.__class__.__name__[:-7]  # Without the Creator

    def deform_name(self):
        return f'{self.deformer.name()}_seq_frac_{self.shape_frac_from_vgroup}'.replace('.', '_')

    def deform_shape(self, fp):
        v, f = self.read_shape(fp)
        deformed = self.deformer.deform(v, f)
        # parts = [v[masks[i][0],:] for i in range(self.deformer.num_expected_deformations())]
        # plot_mesh_montage(vb=parts,strategy='spheres')
        return deformed

    def deform_subject(self, sub):
        banner(title(f'{self.dataset_name()} Dataset :: Subject {sub} :: Deformation {self.deform_name()} Commencing'))
        (self.out_dp / sub).mkdir(exist_ok=True)  # TODO - Presuming this dir structure

        if self.deformer.needs_validation():
            self._deform_subject_validated(sub)
        else:
            self._deform_subject_unvalidated(sub)
        banner(f'Deformation of Subject {sub} - COMPLETED')

    def _deform_subject_validated(self, sub):
        vgd, vgd_fp = self._vgroup_dict(sub)
        vgs_todo = [k for k, v in vgd.items() if v < 1]
        total_failures = 0
        for vgi, vg in tqdm(enumerate(vgs_todo), file=sys.stdout, total=len(vgs_todo), unit='vgroup'):

            comp_frac = self._deform_and_locally_save_vgroup(sub, vg)
            if comp_frac >= self.MIN_VGROUP_SUCCESS_FRAC:
                self._transfer_local_vgroup_to_out_dir(sub, vg)

                vgd[vg] = comp_frac  # Save the VGD to local area:
                with open(vgd_fp, 'w') as handle:
                    json.dump(vgd, handle, sort_keys=True, indent=4)

            elif comp_frac >= 0:
                print_error(f'\nWARNING - Deformation success rate for {vg} is below threshold - skipping')
                total_failures += 1
                if total_failures == self.MAX_FAILED_VGROUPS_BEFORE_RESET:
                    self.deformer.reset()
            else:  # -1 case
                print_warning(f'\nWARNING - Validation Group {vg} has too few shapes - skipping')
        self._print_vgd_statistics(vgd, print_vgd=False)

    def _deform_subject_unvalidated(self, sub):
        for vg in self.vgroups_per_subject(sub):
            vg_dp = self.out_dp / sub / vg  # TODO - Presuming this dir structure
            vg_dp.mkdir(exist_ok=True, parents=True)
            for fp in self.shape_fps_per_vgroup(sub, vg):
                shape_name = fp.with_suffix('')  # TODO - Presuming this name
                deformed = self.deform_shape(fp)
                for i, d in enumerate(deformed):
                    np.savez(vg_dp / f'{shape_name}_{i}.npz', **d)  # TODO - generalize save

    def _deform_and_locally_save_vgroup(self, sub, vg):

        shape_fps = self.shape_fps_per_vgroup(sub, vg)
        if len(shape_fps) < self.MIN_VGROUP_SIZE:
            return -1  # Empty

        if self.shape_frac_from_vgroup != 1:  # Decimate
            requested_number = int(self.shape_frac_from_vgroup * len(shape_fps))
            num_to_take = max(requested_number, self.MIN_VGROUP_SIZE)
            shape_fps = np.random.choice(shape_fps, size=num_to_take, replace=False)

        # Create all needed directories:
        vg_dp = self.out_dp / sub / vg  # TODO - Presuming this dir structure
        assert_new_dir(vg_dp, parents=True)

        completed = 0
        total = len(shape_fps) * self.deformer.num_expected_deformations()
        # Project:
        for fp in shape_fps:
            shape_name = fp.with_suffix('')  # TODO - Presuming this name
            deformed = self.deform_shape(fp)
            i = 0
            for d in deformed:
                if d is not None:
                    np.savez(vg_dp / f'{shape_name}_{i}.npz', **d)  # TODO - generalize save
                    completed += 1
                    i += 1

        return completed / total

    def _transfer_local_vgroup_to_out_dir(self, sub, vg):
        vg_tmp_dp = self.tmp_dp / sub / vg
        vg_out_dp = self.out_dp / sub / vg
        if vg_out_dp.is_dir():
            shutil.rmtree(vg_out_dp)
            time.sleep(2)  # TODO - find something smarter
        shutil.copytree(src=vg_tmp_dp, dst=vg_out_dp)
        shutil.rmtree(vg_tmp_dp, ignore_errors=True)  # Clean up

    def _vgroup_dict(self, sub):
        vgd_fp = self.record_dp / f'{sub}_{self.deform_name()}_validation_dict.json'
        if vgd_fp.is_file():  # Validation Group Dict exists
            with open(vgd_fp, 'r') as handle:
                vgd = json.load(handle)
            self._print_vgd_statistics(vgd)
        else:  # Create it:
            vgd = {vg: 0 for vg in self.vgroups_per_subject(sub)}
            with open(vgd_fp, 'w') as handle:
                json.dump(vgd, handle, sort_keys=True, indent=4)  # Dump as JSON for readability
                print(f'Saved validation group cache for subject {sub} at {vgd_fp}')
        return vgd, vgd_fp

    @staticmethod
    def _print_vgd_statistics(vgd, print_vgd=False):
        # Analysis:
        print('Cache Status:')
        empty, completed, partial = 0, 0, 0
        total = len(vgd)
        for comp_frac in vgd.values():
            empty += (comp_frac == 0)
            completed += (comp_frac == 1)
            partial += (comp_frac != 1 and comp_frac != 0)
        print(f'\t* Completed Validation Groups: {completed}/{total}')
        print(f'\t* Empty Validation Groups: {empty}/{total}')
        print(f'\t* Partial Validation Groups: {partial}/{total}')
        if print_vgd:
            print(json.dumps(vgd, indent=4, sort_keys=True))  # JSON->String

    def subjects(self):
        raise NotImplementedError

    def vgroups_per_subject(self, sub):
        raise NotImplementedError

    def shape_fps_per_vgroup(self, sub, vg):
        raise NotImplementedError


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#

class MixamoCreator(DataCreator):
    RECORD_ROOT = Path(OUTPUT_ROOT)  # Override
    if os.name == 'nt':  # Override
        OUT_ROOT = Path(r'Z:\ShapeCompletion')
    else:  # Presuming Linux
        OUT_ROOT = Path(r"/usr/samba_mount/ShapeCompletion")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(self.deformer, Projection):
            with open(self.COLLAT_DP / 'SMPL_faces.pkl', "rb") as f_file:
                self.f = pickle.load(f_file)  # Already int32
                self.f.flags.writeable = False  # Make this a read-only numpy array

    def subjects(self):
        return tuple(f'0{i}0' for i in range(10))

    def vgroups_per_subject(self, sub):
        fp = self.in_dp / sub
        assert fp.is_dir(), f"Could not find path {fp}"
        return os.listdir(fp)  # glob actually returns a generator

    def shape_fps_per_vgroup(self, sub, vg):
        fp = self.in_dp / sub / vg
        assert fp.is_dir(), f"Could not find path {fp}"
        return list(fp.glob('*.obj'))  # glob actually returns a generator

    def read_shape_for_projection(self, fp):
        # PyRender needs a multiplications of 100 to go smoothly. TODO - Assert this
        v = read_obj_verts(fp).astype('float32') * 100
        v = box_center(v)
        return v, self.f


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#

class SMALCreator(DataCreator):
    MIN_VGROUP_SUCCESS_FRAC = 1
    MIN_VGROUP_SIZE = 0

    def __init__(self, deformer, in_dp):
        super().__init__(deformer, in_dp, 1)

        if isinstance(self.deformer, Projection):
            with open(self.COLLAT_DP / 'SMAL_faces.pkl', "rb") as f_file:
                self.f = pickle.load(f_file)  # Already int32
                self.f.flags.writeable = False  # Make this a read-only numpy array

    def subjects(self):
        return 'cats', 'dogs', 'horses', 'cows', 'hippos'

    def vgroups_per_subject(self, sub):
        fp = self.in_dp / sub
        assert fp.is_dir(), f"Could not find path {fp}"
        return [f.with_suffix('') for f in fp.glob('*')]

    def shape_fps_per_vgroup(self, sub, vg):
        return [self.in_dp / sub / f'{vg}.ply']

    def read_shape_for_projection(self, fp):
        v = read_ply_verts(fp).astype('float32')
        v = box_center(v)
        return v, self.f


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#


if __name__ == '__main__':
    project_smal_main()

# ----------------------------------------------------------------------------------------------------------------------#
#                               Instructions on how to mount server for Linux Machines
# ----------------------------------------------------------------------------------------------------------------------#
# r"/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/Mixamo/Blender/MPI-FAUST"
# Mounting Instructions: [shutil does not support samba soft-links]
# sudo apt install samba
# sudo apt install cifs-utils
# sudo mkdir /usr/samba_mount
# sudo mount -t cifs -o auto,username=mano,uid=$(id -u),gid=$(id -g) //132.68.36.59/data /usr/samba_mount/
# To install CUDA runtime 10.2 on the Linux Machine, go to: https://developer.nvidia.com/cuda-downloads
# And choose the deb(local) version
