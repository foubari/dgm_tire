"""
Epure unimodal VAE model specification.
"""

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from .base_vae import VAE
from .encoder_decoder import Enc, Dec
from .utils import Constants

# Constants
dataSize = torch.Size([1, 64, 32])  # Grayscale, 64x32


class Epure(VAE):
    """Unimodal VAE subclass for Epure experiment."""
    
    def __init__(self, params):
        super(Epure, self).__init__(
            dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,  # prior
            dist.Normal,  # likelihood
            dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,  # posterior
            Enc(
                params.latent_dim_w,
                params.latent_dim_z,
                dist=params.priorposterior,
                nf=params.nf,
                nf_max=params.nf_max,
                cond_dim=getattr(params, 'cond_dim', 0)
            ),  # Encoder model
            Dec(
                params.latent_dim_u,
                nf=params.nf,
                nf_max=params.nf_max // 2,
                cond_dim=getattr(params, 'cond_dim', 0)
            ),  # Decoder model
            params  # params (args from main)
        )
        print(f'likelihood vae {dist.Normal}')
        self._pw_params_aux = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False),
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=True)  # Important: learnable logvar
        ])
        self.modelName = 'Epure-split'
        self.dataSize = dataSize
        self.llik_scaling = 1.
        self.datadir = getattr(params, 'datadir', None)
        self.params = params

    @property
    def pw_params_aux(self):
        """
        Returns: Parameters of prior distribution for modality-specific latent code
        """
        if self.params.priorposterior == 'Normal':
            return self._pw_params_aux[0], F.softplus(self._pw_params_aux[1]) + Constants.eta
        else:  # Laplace
            return self._pw_params_aux[0], F.softmax(self._pw_params_aux[1], dim=-1) * self._pw_params_aux[1].size(-1) + Constants.eta

