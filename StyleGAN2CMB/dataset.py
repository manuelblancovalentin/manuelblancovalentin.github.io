"""
"""


""" Basic modules """
import os
import numpy as np

""" tensorflow """
import tensorflow as tf

""" Custom APS """
from .analysis import AngularPowerSpectroscope

""" CMB maps dataset """
class CMBDataset(object):
    """."""
    def __init__(self, data_dir, maps, transform = None, flip = False, reload = False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.flip = flip

        """ Assert this is an iterable """
        maps = maps if hasattr(maps,'__getitem__') else [maps]

        """ Assert path exists """
        assert(os.path.isdir(data_dir))

        """ Load maps """
        print(f'Loading CMB maps from {data_dir}')
        qu = np.load(os.path.join(data_dir,'sims_qu.npy'))
        q = qu[:,0:1]
        u = qu[:,1:2]
        ekb = np.load(os.path.join(data_dir,'sims_ekb.npy'))
        e = ekb[:,0:1]
        k = ekb[:,1:2]
        b = ekb[:,2:3]

        """ Form maps """
        assert(all([m in ['q','u','e','k','b'] for m in maps]))
        tmp = dict(q = q,
                    u = u,
                    e = e,
                    k = k,
                    b = b)
        maps = {m: tmp[m] for m in maps}

        """ Normalize """
        self.scales = {m: (np.min(maps[m]), np.max(maps[m]), np.ptp(maps[m])) for m in maps}
        maps = {m: (maps[m] - self.scales[m][0])/self.scales[m][2] for m in maps}

        """ Transform to float32 """
        self.maps = {m: np.transpose(maps[m],(0,2,3,1)).astype(np.float32) for m in maps}

        """ 
        Angular Spectra 
        """
        """ Init and store maps factors (for later use when extracting spectra) """
        self.map_factors = {'q': lambda l: 1.,
                       'u': lambda l: 1.,
                       'Q': lambda l: 1.,
                       'U': lambda l: 1.,
                       'e': lambda l: l * (l + 1) / (2 * np.pi),
                       'k': lambda l: 4 / (2 * np.pi) * 1e7,
                       'b': lambda l: l * (l + 1) / (2 * np.pi),
                       'E': lambda l: l * (l + 1) / (2 * np.pi),
                       'B': lambda l: l * (l + 1) / (2 * np.pi)
                       }

        self.ndeg = 5
        self.npix = 128
        """ Extract APS """
        self.scopes = {zn: AngularPowerSpectroscope(self.ndeg,
                                                    self.npix,
                                                    factor=self.map_factors[zn] if zn in self.map_factors else
                                                    self.map_factors[zn])
                       for zn in self.maps}

        if not os.path.isfile(os.path.join(data_dir,'aps_quekb.npy')) or reload:
            self.spectra = {zn: self.scopes[zn].get_spectra(self.maps[zn]) for zn in self.maps}
            np.save(os.path.join(data_dir,'aps_quekb.npy'), self.spectra)
        else:
            self.spectra = np.load(os.path.join(data_dir,'aps_quekb.npy'), allow_pickle=True).item()

    def __len__(self):
        return len(self.maps[list(self.maps.keys())[0]])

    def __getitem__(self, idx):
        if tf.is_tensor(idx):
            idx = idx.tolist()
        return np.concatenate([self.maps[m][idx] for m in self.maps],axis=-1)

    def get_batch(self, num):

        idx = np.random.randint(0, len(self) - 1, num)
        return np.concatenate([self.maps[m][idx] for m in self.maps],axis=-1).astype('float32')

        """ Ignore this for now """
        out = []

        for i in idx:
            out.append(self[i])
            if self.flip and np.random.random() < 0.5:
                out[-1] = np.flip(out[-1], 1)

        return np.array(out).astype('float32')