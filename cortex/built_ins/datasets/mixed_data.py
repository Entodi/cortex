from cortex.plugins import DatasetPlugin, register_plugin
import numpy as np
from scipy import signal

from torch.utils.data import Dataset
import torch

class MixedData(Dataset):
    def __init__(self, n_samples=1024, time_max=8, A=np.array(
        [[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])):
        self._n_samples = n_samples
        self._time_max = time_max
,
        time = np.linspace(0, self._time_max, self._n_samples)
        s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
        s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
        s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal
        S = np.c_[s1, s2, s3]
        S += 0.2 * np.random.normal(size=S.shape)  # Add noise
        S /= S.std(axis=0)  # Standardize data
        X = torch.FloatTensor(np.dot(S, A.T).T)
        #X = X.reshape(1, X.size()[0], X.size()[1])
        self._S = {0: S}
        self._X = {0: X}
        

    def __len__(self):
        return len(self._X) # one sample

    def __getitem__(self, idx):
        return (self._X[idx], self._S[idx])


class MixedDataPLugin(DatasetPlugin):
    sources = ['Mixed']

    def handle(self, source, copy_to_local=False, **transform_args):
        Dataset = self.make_indexing(MixedData)
        train = Dataset()
        test = Dataset()
        dim_images = train[0][0].size()
        input_names = ['images', 'targets', 'index']

        self.add_dataset('train', train)
        self.add_dataset('test', test)
        self.set_input_names(input_names)
        self.set_dims(**dim_images)
        self.set_scale((0, 1))


register_plugin(MixedDataPLugin)

