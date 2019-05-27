import logging

import torch
import torch.nn.functional as F

from cortex.built_ins.models.vae import VAE, VAENetwork
from cortex.built_ins.models.volume_coders import VolumeEncoder, VolumeDecoder
from cortex.plugins import ModelPlugin, register_plugin

logger = logging.getLogger('cortex.volume_vae')

class VolumeVAE(VAE):
    defaults = dict(
        data=dict(
            batch_size=dict(train=2, test=2), inputs=dict(inputs='images')),
        optimizer=dict(optimizer='Adam', learning_rate=1e-4),
        train=dict(save_on_lowest='losses.vae'))

    def __init__(self):
        super().__init__()

        self.encoder = VolumeEncoder(contract=dict(
            kwargs=dict(dim_out='dim_encoder_out')))
        self.decoder = VolumeDecoder(contract=dict(
            kwargs=dict(dim_in='dim_z')))

    def build(self, dim_z=64, dim_encoder_out=1024):
        '''

        Args:
            dim_z: Latent dimension.
            dim_encoder_out: Dimension of the final layer of the decoder before
                             decoding to mu and log sigma.

        '''
        self.encoder.build(encoder_type='vol_convnet')
        self.decoder.build(decoder_type='vol_convnet')

        self.add_noise('Z', dist='normal', size=dim_z)
        encoder = self.nets.encoder
        decoder = self.nets.decoder
        vae = VAENetwork(encoder, decoder, dim_out=dim_encoder_out, dim_z=dim_z)
        self.nets.vae = vae


    def routine(self, inputs, targets, Z, vae_criterion=F.mse_loss,
                beta_kld=1.):
        '''

        Args:
            vae_criterion: Reconstruction criterion.
            beta_kld: Beta scaling for KL term in lower-bound.

        '''

        vae = self.nets.vae
        outputs = vae(inputs)

        try:
            r_loss = vae_criterion(
                outputs, inputs, size_average=False) / inputs.size(0)
        except RuntimeError as e:
            logger.error('Runtime error. This could possibly be due to using '
                         'the wrong encoder / decoder for this dataset. '
                         'If you are using MNIST, for example, use the '
                         'arguments `--encoder_type mnist --decoder_type '
                         'mnist`')
            raise e

        kl = (0.5 * (vae.std**2 + vae.mu**2 - 2. * torch.log(vae.std) -
                     1.).sum(1).mean())


        self.losses.vae = (r_loss + beta_kld * kl)
        self.results.update(KL_divergence=kl.item())


    
    def visualize(self, inputs, targets, Z):
        shape = self.get_dims('x', 'y', 'z', 'c')

        vae = self.nets.vae
        outputs = vae(inputs)

        self.add_image(outputs[:, :, int(shape[0] / 2), :, :], name='reconstruction')
        self.add_image(inputs[:, :, int(shape[0] / 2), :, :], name='ground truth')
        self.add_scatter(vae.mu.data, labels=targets.data, name='latent values')
        self.decoder.visualize(Z)
        

register_plugin(VolumeVAE)
