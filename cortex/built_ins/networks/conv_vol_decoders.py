'''Convolutional decoders

'''

__author__ = 'Alex Fedorov'
__author_email__ = 'eidos92@gmail.com'
__maintainer__ = 'Alex Fedorov'

import logging

import torch.nn as nn

from .modules import View
from .base_network import BaseNet
from .utils import finish_layer_3d


logger = logging.getLogger('cortex.models' + __name__)


def infer_conv_size(w, k, s, p):
    x = (w - k + 2 * p) // s + 1
    return x


class SimpleVolConvDecoder(BaseNet):
    def __init__(self, shape, dim_in=None, initial_layer=None, dim_h=64,
                 nonlinearity='ReLU', output_nonlinearity=None,
                 f_size=4, stride=2, pad=1, n_steps=3, **layer_args):
        super(SimpleVolConvDecoder, self).__init__(
            nonlinearity=nonlinearity, output_nonlinearity=output_nonlinearity)

        dim_h_ = dim_h
        logger.debug('Input shape: {}'.format(shape))
        dim_x_, dim_y_, dim_z_, dim_out_ = shape

        dim_x = dim_x_
        dim_y = dim_y_
        dim_z = dim_z_
        dim_h = dim_h_

        saved_spatial_dimensions = [(dim_x, dim_y, dim_z)]
        for n in range(n_steps):
            dim_x, dim_y, dim_z = self.next_size(dim_x, dim_y, dim_z, f_size, stride, pad)
            dim_x, dim_y, dim_z = self.next_size(dim_x, dim_y, dim_z, f_size, stride, pad)
            saved_spatial_dimensions.append((dim_x, dim_y, dim_z))
            if n < n_steps - 1:
                dim_h *= 2

        dim_out = dim_x * dim_y * dim_z * dim_h

        if initial_layer is not None:
            dim_h_ = [initial_layer, dim_out]
        else:
            dim_h_ = [dim_out]

        self.add_linear_layers(dim_in, dim_h_, **layer_args)

        name = 'reshape to {}x{}x{}x{}'.format(dim_h, dim_x, dim_y, dim_z)
        self.models.add_module(name, View(-1, dim_h, dim_x, dim_y, dim_z))
        
        finish_layer_3d(self.models, name, dim_x, dim_y, dim_z, dim_h,
                        nonlinearity=self.layer_nonlinearity, **layer_args)


        dim_out = dim_h

        for i in range(n_steps):
            dim_in = dim_out

            if i == n_steps - 1:
                pass
            else:
                dim_out = dim_in // 2

            name = 'tconv_s2_({}/{})_{}'.format(dim_in, dim_out, i + 1)
            self.models.add_module(name, nn.ConvTranspose3d(dim_in, dim_out, f_size, stride, pad, bias=False))
            finish_layer_3d(self.models, name, dim_x, dim_y, dim_z, dim_out,
                        nonlinearity=self.layer_nonlinearity, **layer_args)

            name = 'tconv_({}/{})_{}'.format(dim_out, dim_out, i + 1)
            self.models.add_module(
                name, nn.ConvTranspose3d(dim_out, dim_out, f_size, stride, pad,
                                         bias=False))

            finish_layer_3d(self.models, name, dim_x, dim_y, dim_z, dim_out,
                            nonlinearity=self.layer_nonlinearity, **layer_args)
            

        self.models.add_module(name + 'f', nn.Conv3d(
            dim_out, dim_out_, 3, 1, 1, bias=False))

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

        return infer_conv_size(
            dim_x, kx, sx, px), infer_conv_size(
                dim_y, ky, sy, py), infer_conv_size(
                    dim_z, kz, sz, pz)
