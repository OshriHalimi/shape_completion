from pathlib import Path
import numpy as np
import os, sys, inspect, time
# import pyrender
# import trimesh
import h5py
from tqdm import tqdm
from dfaust_query import generate_dfaust_map
from dfaust_utils import write_off, banner, hms_string

# Import render
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, os.path.join(parentdir, 'render', 'lib'))
# import render

# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#

CAM2WORLD = np.array([[0.85408425, 0.31617427, -0.375678, 0.56351697 * 2],
                      [0., -0.72227067, -0.60786998, 0.91180497 * 2],
                      [-0.52013469, 0.51917219, -0.61688, 0.92532003 * 2],
                      [0., 0., 0., 1.]], dtype=np.float32)

# rotate the mesh elevation by 30 degrees
Rx = np.array([[1, 0, 0, 0],
               [0., np.cos(np.pi / 6), -np.sin(np.pi / 6), 0],
               [0, np.sin(np.pi / 6), np.cos(np.pi / 6), 0],
               [0., 0., 0., 1.]], dtype=np.float32)

CAM2WORLD = np.matmul(Rx, CAM2WORLD)

RENDER_INFO = {'Height': 480, 'Width': 640, 'fx': 575, 'fy': 575, 'cx': 319.5, 'cy': 239.5}


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#
def unpack_projected_dataset(dfaust_map, h5py_dir, dump_dir, num_angs=10):
    dfaust_map.num_angs = num_angs
    # A bit hacky
    world2cam_mats = []
    for i_ang, ang in enumerate(np.linspace(0, 2 * np.pi, num_angs)):
        Ry = np.array([[np.cos(ang), 0, -np.sin(ang), 0],
                       [0., 1, 0, 0],
                       [np.sin(ang), 0, np.cos(ang), 0],
                       [0., 0., 0., 1.]], dtype=np.float32)
        world2cam_mats.append(np.linalg.inv(np.matmul(Ry, CAM2WORLD)).astype('float32'))

    dfaust_map.world2cam_mats = world2cam_mats
    # TODO ---------- Render Stub -----------#
    # render.setup(RENDER_INFO)
    # TODO ---------- Render Stub -----------#
    unpack_dataset(dfaust_map, h5py_dir, dump_dir)


def unpack_dataset(dfaust_map, h5py_dir, dump_dir):
    start_t = time.time()
    males = dfaust_map.filter_by_gender('male')
    females = dfaust_map.filter_by_gender('female')
    male_hdf5_fp = Path(h5py_dir) / 'registrations_m.hdf5'
    female_hdf5_fp = Path(h5py_dir) / 'registrations_f.hdf5'

    if males.num_subjects() > 0 and females.num_subjects() > 0:
        if male_hdf5_fp.is_file() and female_hdf5_fp.is_file():
            os.makedirs(dump_dir, exist_ok=True)
        else:
            raise AssertionError(f"Could not find hdf5 files in {Path(h5py_dir).absolute()}")
    else:
        raise AssertionError("Dfaust map is empty")

    # TODO - Add correction to insert size with projections
    print(
        f'Dataset size to unpack: {dfaust_map.disk_size()} [{dfaust_map.num_subjects()} subjects |'
        f' about {dfaust_map.num_sequences()} sequences each | total of {dfaust_map.num_frames()} frames]')
    if dfaust_map.num_angs is not None:
        print(f'Projecting dataset on to {dfaust_map.num_angs} angles')
        # TODO - Remove this
        males.num_angs = dfaust_map.num_angs
        females.num_angs = dfaust_map.num_angs
        females.world2cam_mats = dfaust_map.num_angs
        females.world2cam_mats = dfaust_map.num_angs

    if males.num_subjects() > 0:
        unpack_h5py(males, male_hdf5_fp, dump_dir)
    if females.num_subjects() > 0:
        unpack_h5py(females, female_hdf5_fp, dump_dir)

    print(f"Running time: {hms_string(time.time() - start_t)}")


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#

def unpack_h5py(dfaust_map, h5py_fp, dump_dir):
    seq_cnt = 0
    frame_cnt = 0
    sub_cnt = 0
    # frame_fps = []
    banner(f'Unpacking {h5py_fp.name}')
    time.sleep(0.01)  # For the race condition
    with h5py.File(h5py_fp, 'r') as f:
        for sub in tqdm(dfaust_map.subjects()):
            used_sub = False
            # print(f'{sub.id} ({sub.gender})') # For the creation of the new sub_seq file
            seq_dir = os.path.join(dump_dir, sub.id)  # TODO - Decide on a filename format
            for seq in sub.seq_grp:
                sidseq = sub.id + '_' + seq
                if sidseq in f:
                    used_sub = True
                    seq_cnt += 1
                    verts = f[(sidseq)][()].transpose([2, 0, 1])
                    faces = f[('faces')][()]
                    frame_cnt += verts.shape[0]
                    # print(f'{seq} {verts.shape[0]}') # For the creation of the new sub_seq file
                    tdir = os.path.join(seq_dir, seq)  # TODO - Decide on a filename format
                    os.makedirs(tdir, exist_ok=True)
                    for iv, v in enumerate(verts):
                        # TODO - Decide on a filename format
                        if dfaust_map.num_angs is not None:
                            write_projection_pyrender(os.path.join(tdir, '%05d' % iv), v, faces, dfaust_map)
                        else:
                            write_off(os.path.join(tdir, '%05d.OFF' % iv), v, faces)  # The expensive operation
                        # frame_fps.append(tgt_fp)
                else:
                    print(f'Warning: Sidseq {sidseq} not in {os.path.basename(h5py_fp)}')

            if used_sub:
                sub_cnt += 1
                # print('') # For the creation of the new sub_seq file

    print(f'From {h5py_fp} : Unpacked {frame_cnt} frames from {seq_cnt} sequences and {sub_cnt} subjects')


def write_projection_render(pfp, v, f, map):
    # TODO ---------- Render Stub -----------#
    context = render.SetMesh(v, f)
    for i_ang, _ in enumerate(np.linspace(0, 2 * np.pi, map.num_angs)):
        render.render(context, map.world2cam_mats[i_ang])
        # depth = render.getDepth(RENDER_INFO)
        vindices, _, _ = render.getVMap(context, RENDER_INFO)
        mask = np.unique(vindices)

        # Only kept the mask option
        np.savez(f"{pfp}_mask_{i_ang:03d}", mask=mask)
    # if args.output == 'ply':
    #     write_ply(V[mask, :], os.path.join(args.output_path, f"proj_{i_ang:03d}.ply"))
    # TODO ---------- Render Stub -----------#


def write_projection_pyrender(pfp, v, f, map):  # TODO - Uncompleted
    # RENDER_INFO = {'Height': 480, 'Width': 640, 'fx': 575, 'fy': 575, 'cx': 319.5, 'cy': 239.5}
    mesh = pyrender.Mesh.from_points(v)
    scene = pyrender.Scene()
    scene.add(mesh)
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414))

    # pyrender.Viewer(scene, use_raymond_lighting=True)
    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480, point_size=1.0)
    color, depth = r.render(scene)
    # TODO - Does this have the VMap option?
    # print(depth)


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#


def main_unpack_dataset():
    h5py_dir = Path(__file__).parents[0] / '..' / '..' / '..' / 'shape_completion_WIP' / 'data' / 'dfaust' / 'packed'
    dump_dir = Path(__file__).parents[0] / '..' / '..' / '..' / 'shape_completion_WIP' / 'data' / 'dfaust' / 'unpacked'
    fullmap = generate_dfaust_map()
    unpack_dataset(fullmap, h5py_dir, dump_dir)


def main_unpack_projections_dataset():
    h5py_dir = Path(__file__).parents[0] / '..' / '..' / '..' / '..' / 'shape_completion_WIP' / 'data' / 'dfaust' / 'packed'
    dump_dir = Path(__file__).parents[0] / '..' / '..' / '..' / '..' / 'shape_completion_WIP' / 'data' / 'dfaust' / 'unpacked'
    fullmap = generate_dfaust_map()

    unpack_projected_dataset(fullmap, h5py_dir, dump_dir, num_angs=10)


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    main_unpack_dataset()
