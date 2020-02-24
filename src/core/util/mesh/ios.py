import numpy as np
from util.fs import align_file_extension
from plyfile import PlyData


# TODO - Consider migrating to meshio or PyMesh
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   READ
# ----------------------------------------------------------------------------------------------------------------------#
def read_npz_mask(fp):
    return np.load(fp)["mask"]


def read_obj_verts(fp, nv=6890):
    v = np.zeros((nv, 3))
    v_count = 0
    with open(fp, 'r') as handle:
        for l in handle:
            words = [w for w in l.split(' ') if w != '']
            if words[0] == 'v':
                v[v_count, 0], v[v_count, 1], v[v_count, 2] = float(words[1]), float(words[2]), float(words[3])
                v_count += 1
            elif words[0] == 'f':
                break
    assert v_count == nv, f'Found {v_count} vert. Expected: {nv} verts'
    return v


def read_off_verts(fp):
    vbuf = []
    with open(fp, "r") as f:
        first = f.readline().strip()
        if first != "OFF" and first != "COFF":
            raise (Exception(f"Could not find OFF header for file: {fp}"))

        parameters = f.readline().strip().split()

        if len(parameters) < 2:
            raise (Exception(f"Wrong number of parameters fount at OFF file: {fp}"))

        for i in range(int(parameters[0])):
            xyz = f.readline().split()
            vbuf.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])

    return np.array(vbuf)


def read_off(fp):
    vbuf = []
    fbuf = []
    with open(fp, "r") as f:
        first = f.readline().strip()
        if first != "OFF" and first != "COFF":
            raise (Exception(f"Could not find OFF header for file: {fp}"))

        parameters = f.readline().strip().split()

        if len(parameters) < 2:
            raise (Exception(f"Wrong number of parameters fount at OFF file: {fp}"))

        for i in range(int(parameters[0])):
            xyz = f.readline().split()
            vbuf.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])

        for i in range(int(parameters[1])):
            inds = f.readline().split()
            fbuf.append([int(inds[1]), int(inds[2]), int(inds[3])])

    return np.array(vbuf), np.array(fbuf)


def read_ply(fp):
    with open(fp, 'rb') as f:
        plydata = PlyData.read(f)

    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    v = np.column_stack((x, y, z))
    if 'red' in plydata['vertex']:
        r = plydata['vertex']['red']
        g = plydata['vertex']['green']
        b = plydata['vertex']['blue']
        rgb = np.column_stack((r, g, b))
    else:
        rgb = None
    f = np.stack(plydata['face']['vertex_indices'])
    return v, f, rgb


# ----------------------------------------------------------------------------------------------------------------------#
#                                                   WRITE
# ----------------------------------------------------------------------------------------------------------------------#

def write_off(fp, v, f=None):
    fp = align_file_extension(fp, 'off')
    str_v = [f"{vv[0]} {vv[1]} {vv[2]}\n" for vv in v]
    if f is not None:
        str_f = [f"3 {ff[0]} {ff[1]} {ff[2]}\n" for ff in f]
    else:
        str_f = []

    with open(fp, 'w') as meshfile:
        meshfile.write(f'OFF\n{len(str_v)} {len(str_f)} 0\n{"".join(str_v)}{"".join(str_f)}')


def write_obj(fp, v, f=None):
    fp = align_file_extension(fp, 'obj')
    str_v = [f"v {vv[0]} {vv[1]} {vv[2]}\n" for vv in v]
    if f is not None:
        # Faces are 1-based, not 0-based in obj files
        str_f = [f"f {ff[0]} {ff[1]} {ff[2]}\n" for ff in f + 1]
    else:
        str_f = []

    with open(fp, 'w') as meshfile:
        meshfile.write(f'{"".join(str_v)}{"".join(str_f)}')


# ----------------------------------------------------------------------------------------------------------------------#
#                                                   TODO - Integrate
# ----------------------------------------------------------------------------------------------------------------------#

def write_ply(fp, v, f, n, clrs):
    fp = align_file_extension(fp, 'ply')
    str_vertices = ["{} {} {}".format(v[0], v[1], v[2]) for v in v]
    str_indices = ["3 {} {} {}\n".format(i[0], i[1], i[2]) for i in f]
    str_normals = ["{} {} {}".format(n[0], n[1], n[2]) for n in n]
    # no transparency, alpha = 255
    str_colors = ["{} {} {}".format(c[0], c[1], c[2]) for c in clrs]

    str_vertices = ["{} {} {}\n".format(str_vertices[i], str_normals[i], str_colors[i]) for i in range(len(v))]

    with open(fp, "w") as meshfile:
        meshfile.write('''ply
format ascii 1.0
comment VCGLIB generated
element vertex {0}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
element face {1}
property list uchar int vertex_indices
end_header
{2}
{3}
'''.format(len(str_vertices), len(str_indices), ''.join(str_vertices), ''.join(str_indices)))


# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Tester Functions
# ----------------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    pass
