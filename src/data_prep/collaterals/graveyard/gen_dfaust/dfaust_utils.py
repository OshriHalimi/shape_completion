import glob
import torch
from torch.utils.data import Dataset
import os
from math import log2


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#
def banner(text=None, ch='=', length=78):
    if text is not None:
        spaced_text = ' %s ' % text
    else:
        spaced_text = ''
    print('\n', spaced_text.center(length, ch))


def file_size(size):
    _suffixes = ['bytes', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB']
    # determine binary order in steps of size 10
    # (coerce to int, // still returns a float)
    order = int(log2(size) / 10) if size else 0
    # format file size
    # (.4g results in rounded numbers for exact matches and max 3 decimals,
    # should never resort to exponent values)
    return '{:.4g} {}'.format(size / (1 << (order * 10)), _suffixes[order])


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#
def write_mesh_as_obj(fname, verts, faces):
    with open(fname, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


# returns a list of vertices and a list of triangles (both represented as numpy arrays)
def read_off(off_file):
    vertexBuffer = []
    indexBuffer = []
    with open(off_file, "r") as modelfile:
        first = modelfile.readline().strip()
        if first != "OFF":
            raise (Exception("not a valid OFF file ({})".format(first)))

        parameters = modelfile.readline().strip().split()

        if len(parameters) < 2:
            raise (Exception("OFF file has invalid number of parameters"))

        for i in range(int(parameters[0])):
            coordinates = modelfile.readline().split()
            vertexBuffer.append([float(coordinates[0]), float(coordinates[1]), float(coordinates[2])])

        for i in range(int(parameters[1])):
            indices = modelfile.readline().split()
            indexBuffer.append([int(indices[1]), int(indices[2]), int(indices[3])])

    return np.array(vertexBuffer), np.array(indexBuffer)

# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Untested Functions
# ----------------------------------------------------------------------------------------------------------------------#


# receives a list of vertices and a list of indices (both as numpy arrays)
def write_off(output_file, vertices, indices):
    '''vertices and indices are lists of strings'''

    # converts indices and vertices to a string representation
    str_vertices = ["{} {} {}\n".format(v[0], v[1], v[2]) for v in vertices]
    str_indices = ["3 {} {} {}\n".format(i[0], i[1], i[2]) for i in indices]
    with open(output_file, 'w') as meshfile:
        meshfile.write(
            '''OFF
            %d %d 0
            %s%s
            ''' % (len(str_vertices), len(str_indices), "".join(str_vertices), "".join(str_indices)))


def write_uv_PLY(output_file, vertices, indices, uv):
    str_vertices = ["{} {} {}\n".format(v[0], v[1], v[2]) for v in vertices]
    str_indices = ["3 {} {} {} 6 {} {} {} {} {} {}\n".format(i[0], i[1], i[2],
                                                             uv[i[0]][0], uv[i[0]][1], uv[i[1]][0], uv[i[1]][1],
                                                             uv[i[2]][0], uv[i[2]][1]) for i in indices]
    # str_uv = ["{} {} {}".format(n[0], n[1], n[2]) for n in normals]

    # str_vertices = [ "{} {}\n".format(str_vertices[i], str_normals[i]) for i in range(len(vertices)) ]

    with open(output_file, "w") as meshfile:
        meshfile.write('''ply
    format ascii 1.0
    comment VCGLIB generated
    element vertex {0}
    property float x
    property float y
    property float z
    element face {1}
    property list uchar int vertex_indices
    property list uchar float texcoord
    end_header
{2}
{3}
'''.format(len(str_vertices), len(str_indices), ''.join(str_vertices), ''.join(str_indices)))


def write_PLY(output_file, vertices, indices, normals):
    str_vertices = ["{} {} {}".format(v[0], v[1], v[2]) for v in vertices]
    str_indices = ["3 {} {} {}\n".format(i[0], i[1], i[2]) for i in indices]
    str_normals = ["{} {} {}".format(n[0], n[1], n[2]) for n in normals]

    str_vertices = ["{} {}\n".format(str_vertices[i], str_normals[i]) for i in range(len(vertices))]

    with open(output_file, "w") as meshfile:
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
    element face {1}
    property list uchar int vertex_indices
    end_header
{2}
{3}
'''.format(len(str_vertices), len(str_indices), ''.join(str_vertices), ''.join(str_indices)))


def write_PLY(output_file, vertices, indices, normals, colors):
    str_vertices = ["{} {} {}".format(v[0], v[1], v[2]) for v in vertices]
    str_indices = ["3 {} {} {}\n".format(i[0], i[1], i[2]) for i in indices]
    str_normals = ["{} {} {}".format(n[0], n[1], n[2]) for n in normals]
    # no transparency, alpha = 255
    str_colors = ["{} {} {}".format(c[0], c[1], c[2]) for c in colors]

    str_vertices = ["{} {} {}\n".format(str_vertices[i], str_normals[i], str_colors[i]) for i in range(len(vertices))]

    with open(output_file, "w") as meshfile:
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


# returns a list of numpy arrays
def read_points(filename: str):
    points = []
    with open(filename) as myfile:
        file_lines = myfile.readlines()
        for line in file_lines:
            content = line.split()
            content = [float(n) for n in content]
            # each element is a numpy array
            points.append(content)
    return np.array(points)


# points is a list of numpy arrays
def write_points(points: list, filename: str):
    if len(points) == 0:
        return None
    with open(filename, "w") as myfile:
        for point in points:
            if len(point) == 2:
                myfile.write("{} {}\n".format(point[0], point[1]))
            elif len(point) == 3:
                myfile.write("{} {} {}\n".format(point[0], point[1], point[3]))
            else:
                raise Exception("Points should have dimension 2 or 3")


import numpy as np
import skimage.io as sio


def LoadOff(model_path):
    lines = [l.strip() for l in open(model_path)]
    words = [int(i) for i in lines[1].split(' ')]
    vn = words[0]
    fn = words[1]
    vertices = np.zeros((vn, 3), dtype='float32')
    faces = np.zeros((fn, 3), dtype='int32')
    for i in range(2, 2 + vn):
        vertices[i - 2] = [float(w) for w in lines[i].split(' ')]
    for i in range(2 + vn, 2 + vn + fn):
        digits = [int(w) for w in lines[i].split(' ')]
        if digits[0] != 3:
            print('cannot parse...')
            exit(0)
        faces[i - 2 - vn] = digits[1:]
    return vertices, faces


def LoadTextureOBJ(model_path):
    vertices = []
    vertex_textures = []
    vertex_normals = []
    faces = []
    face_mat = []
    face_textures = []
    face_normals = []
    lines = [l.strip() for l in open(model_path)]
    materials = {}
    kdmap = []
    mat_idx = -1
    filename = model_path.split('/')[-1]
    file_dir = model_path[:-len(filename)]

    for l in lines:
        words = [w for w in l.split(' ') if w != '']
        if len(words) == 0:
            continue

        if words[0] == 'mtllib':
            model_file = model_path.split('/')[-1]
            mtl_file = model_path[:-len(model_file)] + words[1]
            mt_lines = [l.strip() for l in open(mtl_file) if l != '']
            for mt_l in mt_lines:
                mt_words = [w for w in mt_l.split(' ') if w != '']
                if (len(mt_words) == 0):
                    continue
                if mt_words[0] == 'newmtl':
                    key = mt_words[1]
                    materials[key] = np.array([[[0, 0, 0]]]).astype('uint8')
                if mt_words[0] == 'Kd':
                    materials[key] = np.array(
                        [[[float(mt_words[1]) * 255, float(mt_words[2]) * 255, float(mt_words[3]) * 255]]]).astype(
                        'uint8')
                if mt_words[0] == 'map_Kd':
                    if mt_words[1][0] != '/':
                        img = sio.imread(file_dir + mt_words[1])
                    else:
                        img = sio.imread(mt_words[1])
                    if len(img.shape) == 2:
                        img = np.dstack((img, img, img))
                    elif img.shape[2] >= 4:
                        img = img[:, :, 0:3]
                    materials[key] = img

        if words[0] == 'v':
            vertices.append([float(words[1]), float(words[2]), float(words[3])])
        if words[0] == 'vt':
            vertex_textures.append([float(words[1]), float(words[2])])
        if words[0] == 'vn':
            vertex_normals.append([float(words[1]), float(words[2]), float(words[3])])
        if words[0] == 'usemtl':
            mat_idx = len(kdmap)

            kdmap.append(materials[words[1]])

        if words[0] == 'f':
            f = []
            ft = []
            fn = []
            for j in range(3):
                w = words[j + 1].split('/')[0]
                wt = words[j + 1].split('/')[1]
                wn = words[j + 1].split('/')[2]
                f.append(int(w) - 1)
                ft.append(int(wt) - 1)
                fn.append(int(wn) - 1)
            faces.append(f)
            face_textures.append(ft)
            face_normals.append(fn)
            face_mat.append(mat_idx)
    F = np.array(faces, dtype='int32')
    V = np.array(vertices, dtype='float32')
    V = (V * 0.5).astype('float32')
    VN = np.array(vertex_normals, dtype='float32')
    VT = np.array(vertex_textures, dtype='float32')
    FT = np.array(face_textures, dtype='int32')
    FN = np.array(face_normals, dtype='int32')
    face_mat = np.array(face_mat, dtype='int32')

    return V, F, VT, FT, VN, FN, face_mat, kdmap
