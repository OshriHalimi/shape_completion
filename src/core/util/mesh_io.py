import numpy as np


# ----------------------------------------------------------------------------------------------------------------------#
#                                                   READ
# ----------------------------------------------------------------------------------------------------------------------#
def read_npz_mask(fp):
    return np.load(fp)["mask"]


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


# ----------------------------------------------------------------------------------------------------------------------#
#                                                   WRITE
# ----------------------------------------------------------------------------------------------------------------------#

def write_off(fp, v, f=None):
    if f is None:
        f = []
    str_v = [f"{vv[0]} {vv[1]} {vv[2]}\n" for vv in v]
    str_f = [f"3 {ff[0]} {ff[1]} {ff[2]}\n" for ff in f]
    with open(fp, 'w') as meshfile:
        meshfile.write(f'OFF\n{len(str_v)} {len(str_f)} 0\n{"".join(str_v)}{"".join(str_f)}')


def write_obj(fp, v, f=None):
    if f is None:
        f = []
    else:
        f += 1  # Faces are 1-based, not 0-based in obj files
    str_v = [f"v {vv[0]} {vv[1]} {vv[2]}\n" for vv in v]
    str_f = [f"f {ff[0]} {ff[1]} {ff[2]}\n" for ff in f]
    with open(fp, 'w') as meshfile:
        meshfile.write(f'{"".join(str_v)}{"".join(str_f)}')


# ----------------------------------------------------------------------------------------------------------------------#
#                                                   TODO - Integrate
# ----------------------------------------------------------------------------------------------------------------------#

def write_ply(fp, v, f, n, clrs):
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
