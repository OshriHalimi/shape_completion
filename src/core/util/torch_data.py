from torch.utils.data import DataLoader
from copy import deepcopy
from tensorboard import default,program
import logging


def determine_worker_num(num_examples, batch_size):
    import psutil
    num_batch_runs = int(num_examples / batch_size)
    if num_batch_runs < 10:  # Very small amount of runs
        return 0
    else:
        cpu_cnt = psutil.cpu_count(logical=False)
        if batch_size < cpu_cnt:
            return int(batch_size / 2)
        else:
            return int(cpu_cnt / 2)


def loader_ids(loader):
    return list(loader.batch_sampler.sampler.indices)


def exact_num_loader_obj(loader):
    # This is more exact than len(loader)*batch_size - Seeing we don't round up the last batch to batch_size
    return len(loader.dataset)


# ----------------------------------------------------------------------------------------------------------------------
#                                               Argsparse Extension
# ----------------------------------------------------------------------------------------------------------------------
def none_or_str(value):
    if value == 'None':
        return None
    return value  # This is str by default


def none_or_int(value):
    if value == 'None':
        return None
    return int(value)


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
class ReconstructableDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._recon_table = None

    def init_recon_table(self, table):
        self._recon_table = deepcopy(table)

    def recon_table(self):
        return deepcopy(self._recon_table)
