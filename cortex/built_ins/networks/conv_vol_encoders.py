''' Volumetric Convolutional Encoder
'''

__author__ = 'Alex Fedorov'
__author_email__ = 'eidos92@gmail.com'
__maintainer__ = 'Alex Fedorov'

import logging

import torch.nn as nn
import torch.nn.functional as F
from .SpectralNormLayer import SNConv3d, SNLinear

from .modules import View
from .base_network import BaseNet
from .utils import finish_layer_3d
from .convnets import infer_conv_size

logger = logging.getLogger('cortex.arch' + __name__)

class SimpleVolConvEncoder(BaseNet):
    def __init__(self, shape, dim_out=None, dim_h=64,
                 fully_connected_layers=None, nonlinearity='ReLU',
                 output_nonlinearity=None, f_size=4,
                 stride=2, pad=1, min_dim=4, n_steps=None, normalize_input=False,
                 spectral_norm=False, last_conv_nonlinearity=True,
                 last_batchnorm=True, **layer_args):
        super(SimpleVolConvEncoder, self).__init__(
            nonlinearity=nonlinearity, output_nonlinearity=output_nonlinearity)

        Conv3d = SNConv3d if spectral_norm else nn.Conv3d
        Linear = SNLinear if spectral_norm else nn.Linear

        dim_out_ = dim_out
        fully_connected_layers = fully_connected_layers or []
        if isinstance(fully_connected_layers, int):
            fully_connected_layers = [fully_connected_layers]
        
        logger.debug('Input shape: {}'.format(shape))
        dim_x, dim_y, dim_z, dim_in = shape

        if isinstance(dim_h, list):
            n_steps = len(dim_h)

        if normalize_input:
            self.models.add_module('initial_bn', nn.BatchNorm3d(dim_in))

        i = 0
        logger.debug('Input size: {},{},{}'.format(dim_x, dim_y, dim_z))
        while ((dim_x >= min_dim and dim_y >= min_dim and dim_z >= min_dim) and
               (i < n_steps if n_steps else True)):
            if i == 0:
                if isinstance(dim_h, list):
                    dim_out = dim_h[0]
                else:
                    dim_out = dim_h
            else:
                dim_in = dim_out
                if isinstance(dim_h, list):
                    dim_out = dim_h[i]
                else:
                    dim_out = dim_in * 2
            conv_args = dict((k, v) for k, v in layer_args.items())
            name = 'conv_({}/{})_{}'.format(dim_in, dim_out, i + 1)
            self.models.add_module(
                name, Conv3d(dim_in, dim_out, f_size, stride, pad, bias=False))
            dim_x, dim_y, dim_z = self.next_size(dim_x, dim_y, dim_z, f_size, stride, pad)

            is_last_layer = not((dim_x >= min_dim and dim_y >= min_dim and dim_z >= min_dim) and
                                (i < n_steps if n_steps else True))

            if is_last_layer:
                if not(last_conv_nonlinearity):
                    nonlinearity = None
                else:
                    nonlinearity = self.layer_nonlinearity

                if not(last_batchnorm):
                    conv_args['batch_norm'] = False

            finish_layer_3d(
                self.models, name, dim_x, dim_y, dim_z, dim_out,
                nonlinearity=nonlinearity, **conv_args)
            
            
            name = 'conv_s2_({}/{})_{}'.format(dim_out, dim_out, i + 1)
            self.models.add_module(
                name, Conv3d(dim_out, dim_out, f_size, 3, 0, bias=False))
            dim_x, dim_y, dim_z = self.next_size(dim_x, dim_y, dim_z, f_size, 3, 0)
            finish_layer_3d(
                self.models, name, dim_x, dim_y, dim_z, dim_out,
                nonlinearity=nonlinearity, **conv_args)
            
            logger.debug('Output size: {},{},{}'.format(dim_x, dim_y, dim_z))
            i += 1

        if len(fully_connected_layers) == 0 and dim_out_ is None:
            return

        dim_out__ = dim_out
        dim_out = dim_x * dim_y * dim_z * dim_out

        name = 'final_reshape_{}x{}x{}x{}to{}'.format(dim_x, dim_y, dim_z, dim_out__, dim_out)
        self.models.add_module(name, View(-1, dim_out))

        dim_out = self.add_linear_layers(dim_out, fully_connected_layers,
                                         Linear=Linear, **layer_args)
        self.add_output_layer(dim_out, dim_out_, Linear=Linear)

    def next_size(self, dim_x, dim_y, dim_z, k, s, p):
        if isinstance(k, int):
            kx, ky, kz = (k, k, k)
        else:
            kx, ky, kz = k

        if isinstance(s, int):
            sx, sy, sz = (s, s, s)
        else:
            sx, sy, sz = s

        if isinstance(p, int):
            px, py, pz = (p, p, p)
        else:
            px, py, pz = p
        return infer_conv_size(dim_x, kx, sx, px), infer_conv_size(
            dim_y, ky, sy, py), infer_conv_size(dim_z, kz, sz, pz)