'''Volume classifier model

'''

__author__ = 'Alex Fedorov'
__author_email__ = 'eidos92@gmail.com'
__maintainer__ = 'Alex Fedorov'


from cortex.built_ins.models.classifier import SimpleClassifier
from cortex.plugins import (register_plugin, ModelPlugin)
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import update_encoder_args

class VolumeClassification(SimpleClassifier):
    '''Basic image classifier.

    Classifies images using standard convnets.

    '''

    defaults = dict(
        data=dict(batch_size=4, inputs=dict(inputs='images')),
        optimizer=dict(optimizer='Adam', learning_rate=1e-3),
        train=dict(epochs=200, save_on_best='losses.classifier'))

    def build(self, classifier_type='vol_convnet',
              classifier_args=dict(dropout=0.1, fully_connected_layers=[4096, 4096]), Encoder=None):
        '''Builds a simple image classifier.

        Args:
            classifier_type (str): Network type for the classifier.
            classifier_args: Classifier arguments. Can include dropout,
            batch_norm, layer_norm, etc.

        '''
        classifier_args = classifier_args or {}

        shape = self.get_dims('x', 'y', 'z', 'c')
        dim_l = self.get_dims('labels')

        Encoder_, args = update_encoder_args(
            shape, model_type=classifier_type, encoder_args=classifier_args)
        Encoder = Encoder or Encoder_
        args.update(**classifier_args)

        classifier = Encoder(shape, dim_out=dim_l, **args)
        print (classifier)
        self.nets.classifier = classifier

    def visualize(self, images, inputs, targets):
        predicted = self.predict(inputs)
        shape = self.get_dims('x', 'y', 'z', 'c')
        self.add_image(images.data[:, :, int(shape[0] / 2), :, :], labels=(targets.data, predicted.data),
                       name='gt_pred')

register_plugin(VolumeClassification)
