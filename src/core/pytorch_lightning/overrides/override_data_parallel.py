"""
.. warning:: `override_data_parallel` module has been renamed to `data_parallel` since v0.6.0
 and will be removed in v0.8.0
"""

import warnings

warnings.warn("`override_data_parallel` module has been renamed to `data_parallel` since v0.6.0"
              " and will be removed in v0.8.0", DeprecationWarning)

from pytorch_lightning.overrides.data_parallel import (  # noqa: E402
    get_a_var, parallel_apply, LightningDataParallel, LightningDistributedDataParallel)
