from cortex.plugins import ModelPlugin
from cortex.built_ins.models.utils import update_encoder_args, update_decoder_args, ms_ssim
import torch.nn.functional as F

class VolumeEncoder(ModelPlugin):

    def build(self,
              dim_out=None,
              encoder_type: str = 'vol_convnet',
              encoder_args=dict(fully_connected_layers=1024),
              Encoder=None):
        x_shape = self.get_dims('x', 'y', 'z', 'c')
        Encoder_, encoder_args = update_encoder_args(
            x_shape, model_type=encoder_type, encoder_args=encoder_args)
        Encoder = Encoder or Encoder_
        encoder = Encoder(x_shape, dim_out=dim_out, **encoder_args)
        print (encoder)
        self.nets.encoder = encoder

    def encode(self, inputs, **kwargs):
        return self.nets.encoder(inputs, **kwargs)

    def visualize(self, inputs, targets):
        Z = self.encode(inputs)
        if targets is not None:
            targets = targets.data
        self.add_scatter(Z.data, labels=targets, name='latent values')

class VolumeDecoder(ModelPlugin):

    def build(self,
              dim_in=None,
              decoder_type: str = 'vol_convnet',
              decoder_args=dict(output_nonlinearity='tanh'),
              Decoder=None):
        x_shape = self.get_dims('x', 'y', 'z', 'c')
        Decoder_, decoder_args = update_decoder_args(
            x_shape, model_type=decoder_type, decoder_args=decoder_args)
        Decoder = Decoder or Decoder_
        decoder = Decoder(x_shape, dim_in=dim_in, **decoder_args)
        print (decoder)
        self.nets.decoder = decoder

    def routine(self, inputs, Z, decoder_crit=F.mse_loss):
        X = self.decode(Z)
        self.losses.decoder = decoder_crit(X, inputs) / inputs.size(0)
        msssim = ms_ssim(inputs, X)
        self.results.ms_ssim = msssim.item()

    def decode(self, Z):
        return self.nets.decoder(Z)

    def visualize(self, Z):
        # TODO: need specific vizualization for 3D volumes
        shape = self.get_dims('x', 'y', 'z', 'c')
        gen = self.decode(Z)
        self.add_image(gen[:, :, int(shape[0] / 2), :, :], name='generated')
