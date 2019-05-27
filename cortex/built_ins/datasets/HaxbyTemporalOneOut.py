from cortex.plugins import DatasetPlugin, register_plugin
from cortex.built_ins.datasets.fmri_dataload import ImageFolder as NII_ImageFolder

import torchvision.transforms as transforms
from cortex.built_ins.datasets.HaxbySlicedOneOut import unit_interval_normalization

class HaxbyTemporalOneOutPlugin(DatasetPlugin):
    sources = ['HaxbyTemporalOneOut']

    def handle(self, source, copy_to_local=False, **transform_args):
        Dataset = self.make_indexing(NII_ImageFolder)
        data_path = self.get_path(source)

        if isinstance(data_path, dict):
            train_path = data_path['train']
            test_path = data_path['test']
            if copy_to_local:
                train_path = self.copy_to_local_path(train_path)
                test_path = self.copy_to_local_path(test_path)
        elif isinstance(data_path, (tuple, list)):
            train_path, test_path = data_path
            if copy_to_local:
                train_path = self.copy_to_local_path(train_path)
                test_path = self.copy_to_local_path(test_path)
        else:
            train_path = data_path
            if copy_to_local:
                train_path = self.copy_to_local_path(train_path)
            test_path = data_path

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: unit_interval_normalization(x))
        ])

        train_set = Dataset(root=train_path, transform=transform)
        test_set = Dataset(root=test_path, transform=transform)
        input_names = ['images', 'targets', 'index']

        dim_t, dim_c, dim_x, dim_y, dim_z = train_set[0][0].size()
        dim_l = len(train_set.classes)

        dims = dict(x=dim_x, y=dim_y, z=dim_z, c=dim_c, t=dim_t, labels=dim_l)

        self.add_dataset('train', train_set)
        self.add_dataset('test', test_set)
        self.set_input_names(input_names)
        self.set_dims(**dims)

        self.set_scale((0, 1))

register_plugin(HaxbyTemporalOneOutPlugin)
