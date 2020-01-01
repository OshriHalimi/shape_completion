import numpy as np
import scipy
from sklearn.preprocessing import normalize


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#

def index_sparse(columns, indices, data=None):
    """
    Return a sparse matrix for which vertices are contained in which faces.
    A data vector can be passed which is then used instead of booleans
    """
    indices = np.asanyarray(indices)
    columns = int(columns)
    row = indices.reshape(-1)
    col = np.tile(np.arange(len(indices)).reshape((-1, 1)), (1, indices.shape[1])).reshape(-1)

    shape = (columns, len(indices))
    if data is None:
        data = np.ones(len(col), dtype=np.bool)
    # assemble into sparse matrix
    matrix = scipy.sparse.coo_matrix((data, (row, col)), shape=shape, dtype=data.dtype)

    return matrix

def normc(mat):
    return normalize(mat, norm='l2', axis=0)


def normr(mat):
    return normalize(mat, norm='l2', axis=1)


def normv(vec):
    return normalize(vec, norm='l2')
# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#
