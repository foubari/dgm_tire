"""
Epure experiment MMVAEplus model specifications.
"""

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from .model import MMVAEplus
from .vae import Epure
from .utils import Constants

# Components: 5 instead of 6 (removed 'gi')
components = ['group_nc', 'group_km', 'bt', 'fpu', 'tpc']


class MMVAEplusEpure(MMVAEplus):
    """
    MMVAEplus subclass for Epure Experiment.
    """
    
    def __init__(self, params):
        # Create 5 VAEs (one per component) instead of 6
        super(MMVAEplusEpure, self).__init__(
            dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,
            params,
            Epure, Epure, Epure, Epure, Epure  # 5 VAEs
        )

        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_z), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim_z), requires_grad=False)  # logvar
        ])
        self._pw_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False)  # logvar
        ])
        self.modelName = 'MMVAEplus_Epure'

        # Fix model names for individual models to be saved
        for comp, vae in zip(components, self.vaes):
            vae.modelName = 'VAE_' + comp

        self.datadir = getattr(params, 'datadir', None)
        self.params = params

    @property
    def pz_params(self):
        """
        Returns: Parameters of prior distribution for shared latent code
        """
        if self.params.priorposterior == 'Normal':
            return self._pz_params[0], F.softplus(self._pz_params[1]) + Constants.eta
        else:  # Laplace
            return self._pz_params[0], F.softmax(self._pz_params[1], dim=-1) * self._pz_params[1].size(-1) + Constants.eta

    @property
    def pw_params(self):
        """
        Returns: Parameters of prior distribution for modality-specific latent code
        """
        if self.params.priorposterior == 'Normal':
            return self._pw_params[0], F.softplus(self._pw_params[1]) + Constants.eta
        else:  # Laplace
            return self._pw_params[0], F.softmax(self._pw_params[1], dim=-1) * self._pw_params[1].size(-1) + Constants.eta

