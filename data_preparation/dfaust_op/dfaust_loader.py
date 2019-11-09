import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import os
from pathlib import Path
import numpy as np
import os
import trimesh
from dfaust_query import generate_dfaust_map
from dfaust_utils import read_off, banner


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#
class DFaustDataset(Dataset):

    def __init__(self, dir, dmap):

        dir = Path(dir)
        self.meshes = []
        for sub in dmap.subjects():
            for seq in sub.seq_grp:  # Unencoded

                tgt_dir = dir / sub.id / seq
                if tgt_dir.is_dir():
                    glob_pat = str((tgt_dir / '*.OFF').absolute())
                    found_meshes = glob.glob(glob_pat)
                    if not found_meshes:
                        print(f'Warning: target directory {tgt_dir} does not hold OFF files')
                    else:
                        self.meshes += found_meshes
                else:
                    print(f'Warning: target directory {tgt_dir} does not exist')

        print(f'Found {len(self.meshes)} mesh files')

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, idx):
        # v = trimesh.load(self.meshes[idx])
        v, _ = read_off(self.meshes[idx])  # TODO - This could be made faster
        v = torch.from_numpy(v).cuda()
        return v


def get_dfaust_loader(batch_size=10, shuffle=True, device='cuda'):
    if device.lower() == 'cuda' and torch.cuda.is_available():
        num_workers, pin_memory = 1, True
    else:
        print('Warning: Did not find working GPU - Loading dataset on CPU')
        num_workers, pin_memory = 4, False

    dump_dir = Path() / '..' / '..' / 'data' / 'dfaust' / 'dfaust_unpacked'
    fullmap = generate_dfaust_map()

    ds = DFaustDataset(dir=dump_dir, dmap=fullmap)
    # Iteration returns torch cuda arrays of vertices: Nv x 3
    for vs in ds:
        print(vs.shape)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return dataloader


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    get_dfaust_loader()
