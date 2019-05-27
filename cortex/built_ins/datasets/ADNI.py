from os import path

from cortex.plugins import DatasetPlugin, register_plugin
from cortex.built_ins.datasets.fmri_dataload import ImageFolder as NII_ImageFolder

import torchvision.transforms as transforms
from cortex.built_ins.datasets.HaxbySlicedOneOut import unit_interval_normalization
import numpy as np
import torch

def random_crop(x, shape=[110, 130, 104]):
    #print (x.size())
    x_shape = x.size()[1:]
    coord1 = np.random.randint(11)
    coord2 = np.random.randint(10)
    coord3 = np.random.randint(4)
    ret = x[:,coord1:coord1+shape[0],coord2:coord2+shape[1],coord3:coord3+shape[2]]
    #print (ret.size())
    return ret

def fixed_crop(x, shape=[110, 130, 104]):
    x_shape = x.size()[1:]
    ret = x[:,5:5+shape[0],5:5+shape[1],2:2+shape[2]]
    return ret

def zero_pad_center_crop(x, shape=[128, 128, 128], crop=[110, 128, 104], coords=[11, 10, 4]):
    coord1 = 4
    coord2 = 7
    coord3 = 2
    pad = (np.array(shape) - np.array(crop)) // 2
    pad = ((0,0), (pad[0], pad[0]), (pad[1], pad[1]), (pad[2], pad[2]))
    ret = x[:, coord1:coord1+crop[0],coord2:coord2+crop[1],coord3:coord3+crop[2]]
    ret = np.pad(ret, pad, mode='constant')
    return torch.from_numpy(ret[:, :128, :128, :128])

def zero_pad_random_crop(x, shape=[1, 128, 128, 128], crop=[110, 128, 104], coords=[11, 10, 4]):
    ret = torch.zeros(shape)
    coord1 = np.random.randint(coords[0])
    coord2 = np.random.randint(coords[1])
    coord3 = np.random.randint(coords[2])
    
    c_coord1 = np.random.randint(shape[1] - crop[0])
    c_coord3 = np.random.randint(shape[3] - crop[2])
    ret[:, c_coord1:c_coord1+crop[0], :, c_coord3:c_coord3+crop[2]] = x[:, coord1:coord1+crop[0],coord2:coord2+crop[1],coord3:coord3+crop[2]]
    return ret

def flip(x, axis=1, p=0.5):
    n = np.random.rand()
    ret = x
    if n > p:
        ret = torch.from_numpy(np.flip(ret, axis=axis).copy())
    return ret
    

class ADNIPlugin(DatasetPlugin):
    sources = ['ADNI0', 'ADNI1', 'ADNI2', 'ADNI3', 
        'ADNI4','adni_only0', 'adni_only1', 'adni_only2', 'adni_only3', 'adni_only4']

    def handle(self, source, copy_to_local=False, **transform_args):
        Dataset = self.make_indexing(NII_ImageFolder)
        data_path = self.get_path(source)

        train_path = path.join(data_path, 'train')
        test_path = path.join(data_path, 'val_adni')

        train_transform = transforms.Compose([
            transforms.Lambda(lambda x: zero_pad_center_crop(x)),
            #transforms.Lambda(lambda x: zero_pad_random_crop(x)),
            #transforms.Lambda(lambda x: flip(x, axis=0)),
            #transforms.Lambda(lambda x: flip(x, axis=1)),
            #transforms.Lambda(lambda x: flip(x, axis=2)),
            #transforms.Lambda(lambda x: ToTensor(x)),
            #transforms.Lambda(lambda x: unit_interval_normalization(x))
        ])

        test_transform = transforms.Compose([
            transforms.Lambda(lambda x: zero_pad_center_crop(x)),
            #transforms.Lambda(lambda x: ToTensor(x)),
            #transforms.Lambda(lambda x: unit_interval_normalization(x))
        ])

        print (train_transform)
        print (test_transform)

        train_set = Dataset(root=train_path, transform=train_transform)
        test_set = Dataset(root=test_path, transform=test_transform)
        print (len(train_set), len(test_set))
        input_names = ['images', 'targets', 'index']

        dim_c, dim_x, dim_y, dim_z = train_set[0][0].size()
        print (train_set[0][0].min(), train_set[0][0].max())
        dim_l = len(train_set.classes)
    
        dims = dict(x=dim_x, y=dim_y, z=dim_z, c=dim_c, labels=dim_l)
        print (dims)
        self.add_dataset('train', train_set)
        self.add_dataset('test', test_set)
        self.set_input_names(input_names)
        self.set_dims(**dims)

        self.set_scale((0, 1))

register_plugin(ADNIPlugin)
