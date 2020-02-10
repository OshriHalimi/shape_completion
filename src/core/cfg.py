import pathlib  # PyCharm tip : Use CTRL + SHIFT + f in to detect where these are used

# ----------------------------------------------------------------------------------------------------------------------
#                                           ADDITIONAL LEARNING PARAMS
# ----------------------------------------------------------------------------------------------------------------------
UNIVERSAL_RAND_SEED = 2147483647  # The random seed. Use datetime.now() For a truly random seed
UNIVERSAL_PRECISION = 'float32'  # float64,float32 or float16. PyTorch defaults to float32.
# TODO - This does not propagate to faces. Is this a problem ?
# TODO - VTK does not work with float16 - should we transform the CPU tensors before plot?

NORMAL_MAGNITUDE_THRESH = 10 ** (-6)  # The minimal norm allowed for vertex normals to decide that they are too small
DEF_LR_SCHED_COOLDOWN = 5  # Number of epoches to wait after reducing the step-size. Works only if LR sched is enabled
DEF_MINIMAL_LR = 1e-6  # The smallest learning step allowed with LR sched. Works only if LR sched is enabled
MAX_EPOCHS = 2  # Maximum number of allowed epochs. We usually stop WAY before this due to early stop callback
# ----------------------------------------------------------------------------------------------------------------------
#                                                    COMPLEXITY
# ----------------------------------------------------------------------------------------------------------------------
NON_BLOCKING = True  # Transfer to GPU in a non-blocking method
REPORT_LOSS_PER_BATCH = False  # If True - will output train loss to logger on every batch. Otherwise - on every epoch
# ----------------------------------------------------------------------------------------------------------------------
#                                                      ERROR
# ----------------------------------------------------------------------------------------------------------------------
SUPPORTED_IN_CHANNELS = (3, 6, 12)  # The possible supported input channels - either 3, 6 or 12
DANGEROUS_MASK_THRESH = 100  # The minimal length allowed for mask vertex indices.
# ----------------------------------------------------------------------------------------------------------------------
#                                                   FILE SYSTEM
# ----------------------------------------------------------------------------------------------------------------------
PRIMARY_RESULTS_DIR = (pathlib.Path(__file__).parents[0] / '..' / '..' / 'results').resolve()
PRIMARY_DATA_DIR = (pathlib.Path(__file__).parents[0] / '..' / '..' / 'data').resolve()
SMPL_TEMPLATE_PATH = PRIMARY_DATA_DIR / 'templates' / 'template_color.ply'
SAVE_MESH_AS = 'obj'  # Currently implemented - ['obj','off']
# ----------------------------------------------------------------------------------------------------------------------
#                                                   VISUALIZATION
# ----------------------------------------------------------------------------------------------------------------------
VIS_N_MESH_SETS = 2  # Parallel plot will plot 8 meshes for each mesh set - 4 from train, 4 from vald
VIS_STRATEGY = 'cloud'  # spheres,cloud,mesh  - Choose how to display the meshes
VIS_CMAP = 'summer'  # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
# We use two colors: one for the mask verts [Right end of the spectrum] and one for the rest [Left end of the spectrum].
VIS_SHOW_GRID = False  # Visualzie with grid?
VIS_SMOOTH_SHADING = False  # Smooth out the mesh before visualization?  Applicable only for 'mesh' method
VIS_SHOW_EDGES = False  # Visualize with edges? Applicable only for 'mesh' method
VIS_SHOW_NORMALS = False
