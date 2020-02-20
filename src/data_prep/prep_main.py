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
import random

sys.path.append(os.path.abspath(os.path.join('..', 'core')))
from util.string_op import banner, print_color, title
from util.mesh.io import read_obj_verts
from util.mesh.plot import plot_mesh_montage, plot_mesh
from util.mesh.ops import box_center
from util.fs import assert_new_dir

# ----------------------------------------------------------------------------------------------------------------------#
#                                                       Globals
# ----------------------------------------------------------------------------------------------------------------------#
WORK_DIR = (Path(__file__).parents[0]).resolve()
OUTPUT_DIR = WORK_DIR / 'tmp_outputs'
COLLATERALS_DIR = WORK_DIR / 'collaterals'

if os.name == 'nt':
    MIXAMO_NETWORK_DIR = Path('Z:\ShapeCompletion\Mixamo\Blender\MPI-FAUST')
    MIXAMO_NETWORK_OUT_DIR = Path('Z:\ShapeCompletion\Mixamo')
else:  # Presuming Linux
    MIXAMO_NETWORK_DIR = Path("/usr/samba_mount/ShapeCompletion/Mixamo/Blender/MPI-FAUST")
    # r"/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/Mixamo/Blender/MPI-FAUST"
    MIXAMO_NETWORK_OUT_DIR = Path("/usr/samba_mount/ShapeCompletion/Mixamo")


# Mounting Instructions: [shutil does not support samba soft-links]
# sudo apt install samba
# sudo apt install cifs-utils
# sudo mkdir /usr/samba_mount
# sudo mount -t cifs -o username=mano,uid=$(id -u),gid=$(id -g) //132.68.36.59/data /usr/samba_mount/
# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#


def project_mixamo_main():
    banner('MIXAMO Creation')
    deformer = Projection(num_angles=10, pick_k=2)
    m = MixamoCreator(deformer=deformer, pose_frac_from_sequence=1)
    for sub in m.subjects():
        m.deform_subject(sub=sub, rerun_partial_seqs=True)
        if sub == '050':
            break
        # We can break here, if we want to split the workload to many computers


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#
class DataCreator:
    ds_output_dp = Path(OUTPUT_DIR)
    collat_dp = Path(COLLATERALS_DIR)

    def __init__(self, deformer):
        self.deformer = deformer
        ds_name = self.dataset_name()
        self.tmp_dp = self.ds_output_dp / ds_name / 'tmp'
        self.dump_dp = self.ds_output_dp / ds_name / self.deform_identifier()
        self.tmp_dp.mkdir(parents=True, exist_ok=True)
        self.dump_dp.mkdir(parents=True, exist_ok=True)
        print(f'Local outputs will be located at {self.dump_dp}')

    def dataset_name(self):
        return self.__class__.__name__[:-7]  # Without the Creator

    def deform_identifier(self):
        return self.deformer.name()


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#

class MixamoCreator(DataCreator):
    MIXAMO_SUB_NAMES = tuple(f'0{i}0' for i in range(10))  # General Usage
    LOWEST_COMPLETION_THRESH = 0.5  # Under this, the sequence will not be taken
    MIN_NUMBER_OF_POSES_PER_SEQUENCE = 10  # Will not take the sequence if under this
    PROJ_SCALE_BY = 100  # How much to scale the vertices by for PyRender deformation

    def __init__(self, deformer, pose_frac_from_sequence=1):
        self.pose_frac_from_sequence = pose_frac_from_sequence
        super().__init__(deformer)
        # Create Network Results:
        self.network_full_dp = Path(MIXAMO_NETWORK_DIR)
        self.network_dump_dp = Path(MIXAMO_NETWORK_OUT_DIR) / self.deform_identifier()
        assert self.network_full_dp.is_dir(), f"Cannot find network path {self.network_full_dp}"
        self.network_dump_dp.mkdir(parents=True, exist_ok=True)
        print(f'Network outputs will be located at {self.network_dump_dp}')

        # TODO - Generalize to other forms of deformations
        if isinstance(deformer, Projection):
            with open(self.collat_dp / 'SMPL_template.pkl', "rb") as f_file:
                self.f = pickle.load(f_file)  # Already int32
                self.f.flags.writeable = False  # Make this a read-only numpy array

    def subjects(self):
        return self.MIXAMO_SUB_NAMES

    def deform_identifier(self):
        return f'{self.deformer.name()}_seq_frac_{self.pose_frac_from_sequence}'.replace('.', '_')

    def network_sequence_fp_list_per_subject(self, sub_name):
        fp = self.network_full_dp / sub_name
        assert fp.is_dir(), f"Could not find path {fp}"
        return list(fp.glob('*'))  # glob actually returns a generator

    def network_pose_fp_list_per_sequence(self, sub_name, seq_name):
        fp = self.network_full_dp / sub_name / seq_name
        assert fp.is_dir(), f"Could not find path {fp}"
        return list(fp.glob('*.obj'))  # glob actually returns a generator

    def deform_subject(self, sub, rerun_partial_seqs=True):
        banner(title(f'Mixamo Dataset :: Subject {sub} :: Deformation {self.deform_identifier()} Commencing'))
        lcd, lcd_fp = self._local_cache_dict(sub)
        # lcd_fp = str(lcd_fp) # Atomic write does not support pathlib
        if rerun_partial_seqs:
            seqs_todo = [k for k, v in lcd.items() if v < 1]
            print('Compute Mode: Empty + Partial Sequences')
        else:
            seqs_todo = [k for k, v in lcd.items() if v == 0]
            print('Compute Mode: Empty')

        banner('Computing Workload')
        (self.network_dump_dp / sub).mkdir(exist_ok=True)

        for seqi, seq in tqdm(enumerate(seqs_todo), file=sys.stdout, dynamic_ncols=True,
                              total=len(seqs_todo), unit='sequence'):
            # print(f'{seqi}/{len(seqs_todo)} :: Deforming sequence {seq}')
            comp_frac = self._deform_and_locally_save_sequence(sub, seq)
            if comp_frac >= self.LOWEST_COMPLETION_THRESH:
                # print(f'Transferring sequence {seq} to network location')
                self._transfer_local_deformed_sequence_to_network_area(sub, seq)
                # Save the LCD to local area:
                lcd[seq] = comp_frac
                with open(lcd_fp, 'w') as handle:
                    json.dump(lcd, handle, sort_keys=True, indent=4)
                # print(f'Recorded completion of {seq} with success rate = {comp_frac}')
            elif comp_frac >= 0:
                print_color(f'WARNING - Deformation success rate for {seq} is below threshold - skipping')
            else:  # -1 case
                print_color(f'WARNING - Sequence {seq} has too few sequences - skipping')
            self.deformer._reset()
        banner(f'Deformation of Subject {sub} - COMPLETED')
        self._print_lcd_analysis(lcd, print_lcd=True)

    def _deform_and_locally_save_sequence(self, sub, seq):

        deform_fps = self.network_pose_fp_list_per_sequence(sub, seq)
        if len(deform_fps) < self.MIN_NUMBER_OF_POSES_PER_SEQUENCE:
            return -1  # Empty

        if self.pose_frac_from_sequence != 1:  # Decimate
            requested_number = int(self.pose_frac_from_sequence * len(deform_fps))
            num_to_take = max(requested_number, self.MIN_NUMBER_OF_POSES_PER_SEQUENCE)
            deform_fps = np.random.choice(deform_fps, size=num_to_take, replace=False)

        # Create all needed directories:
        seq_dp = self.dump_dp / sub / seq
        assert_new_dir(seq_dp, parents=True)
        # for fp in deform_fps:
        #     pose = fp.name[:-4]  # Remove obj
        #     dump_dp = seq_dp / pose
        #     assert_new_dir(dump_dp)  # Possibly running over old projction directory
        #     dump_dps.append(dump_dp)

        completed = 0
        total = len(deform_fps) * self.deformer.num_expected_deformations()
        # Project:
        for deform_fp in deform_fps:
            pose = deform_fp.name[:-4]  # Remove obj
            masks = self._deform_pose(deform_fp)
            i = 0
            for mask in masks:
                if mask is not None:
                    completed += 1
                    i += 1
                    np.savez(seq_dp / f'{pose}_{i}_angi_{mask[1]}.npz', mask=mask[0])  # TODO - remove tuple assumption

        return completed / total

    def _deform_pose(self, pose_fp):
        # TODO - Generalize these two lines to other deformations
        v = read_obj_verts(pose_fp) * self.PROJ_SCALE_BY
        # k = self.deformer.num_expected_deformations()  # HACK
        # return [(v, random.randint(0, self.deformer.num_angles)) for _ in range(k)]  # HACK
        v = box_center(v)
        # plot_mesh(v[masks[0][0],:], strategy='spheres', grid_on=True)
        masks = self.deformer.deform(v, self.f)
        # parts = [v[masks[i][0],:] for i in range(self.deformer.num_expected_deformations())]
        # plot_mesh_montage(vb=parts,strategy='spheres')
        return masks

    def _transfer_local_deformed_sequence_to_network_area(self, sub, seq):
        local_dp = self.dump_dp / sub / seq
        network_dp = self.network_dump_dp / sub / seq
        if network_dp.is_dir():
            shutil.rmtree(network_dp)
            time.sleep(2)  # TODO - find something smarter
        shutil.copytree(src=local_dp, dst=network_dp)
        shutil.rmtree(local_dp, ignore_errors=True)  # Clean up

    def _local_cache_dict(self, sub):
        lcd_fp = self.tmp_dp / f'{sub}_{self.deform_identifier()}_validation_dict.json'
        if lcd_fp.is_file():
            with open(lcd_fp, 'r') as handle:
                lcd = json.load(handle)
            self._print_lcd_analysis(lcd)

        else:  # Create it:
            lcd = {seq_fp.name: 0 for seq_fp in self.network_sequence_fp_list_per_subject(sub)}
            with open(lcd_fp, 'w') as handle:
                json.dump(lcd, handle, sort_keys=True, indent=4)  # Dump as JSON for readability
                print(f'Saved validation cache for subject {sub} at {lcd_fp}')
        return lcd, lcd_fp

    def _print_lcd_analysis(self, lcd, print_lcd=False):
        # Analysis:
        print('Cache Status:')
        empty, completed, partial = 0, 0, 0
        total = len(lcd)
        for v in lcd.values():
            empty += (v == 0)
            completed += (v == 1)
            partial += (v != 1 and v != 0)
        print(f'\t* Completed Sequences: {completed}/{total}')
        print(f'\t* Empty Sequences: {empty}/{total}')
        print(f'\t* Partial Sequences: {partial}/{total}')
        if print_lcd:
            print(json.dumps(lcd, indent=4, sort_keys=True))  # JSON->String


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#


if __name__ == '__main__':
    project_mixamo_main()

# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#
