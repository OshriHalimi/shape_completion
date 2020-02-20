from ctypes import *
import numpy as np
import os
from warnings import warn

LIB_PATH = os.path.dirname(os.path.abspath(__file__))
try:
    Render = cdll.LoadLibrary(os.path.join(LIB_PATH, 'libRender.so'))
except OSError as e:
    if e.errno == 8:  # winerror 193
        warn('PyRender does not support Windows')
    else:
        raise e


def setup(info):
    Render.InitializeCamera(info['Width'], info['Height'],
                            c_float(info['fx']), c_float(info['fy']), c_float(info['cx']), c_float(info['cy']))


def set_mesh(V, F):
    handle = Render.SetMesh(c_void_p(V.ctypes.data), c_void_p(F.ctypes.data), V.shape[0], F.shape[0])
    return handle


def render(handle, world2cam):
    Render.SetTransform(handle, c_void_p(world2cam.ctypes.data))
    Render.Render(handle)


def get_depth(info):
    depth = np.zeros((info['Height'], info['Width']), dtype='float32')
    Render.GetDepth(c_void_p(depth.ctypes.data))

    return depth


def get_vmap(handle, info):
    vindices = np.zeros((info['Height'], info['Width'], 3), dtype='int32')
    vweights = np.zeros((info['Height'], info['Width'], 3), dtype='float32')
    findices = np.zeros((info['Height'], info['Width']), dtype='int32')

    Render.GetVMap(handle, c_void_p(vindices.ctypes.data), c_void_p(vweights.ctypes.data),
                   c_void_p(findices.ctypes.data))

    return vindices, vweights, findices


def colorize(VC, vindices, vweights, mask, cimage):
    Render.Colorize(c_void_p(VC.ctypes.data), c_void_p(vindices.ctypes.data), c_void_p(vweights.ctypes.data),
                    c_void_p(mask.ctypes.data), c_void_p(cimage.ctypes.data), vindices.shape[0], vindices.shape[1])


def clear():
    Render.ClearData()

def reset():
    global Render
    del Render
    Render = cdll.LoadLibrary(os.path.join(LIB_PATH, 'libRender.so'))


