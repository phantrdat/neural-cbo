import os
import numpy as np
import torch

from .basefunc import BaseFunc


class GasTransmission_2_3_15_OBJ(BaseFunc):
    def __init__(
        self,
        xsize=10,
        zsize=10,
        transformation="",
        noise_std=0.01,
    ):
        xdim = 2
        zdim = 2

        super(GasTransmission_2_3_15_OBJ, self).__init__(
            xdim, zdim, xsize, zsize, transformation, noise_std=noise_std
        )

        self.module_name = "gas_transmission_2_3_15_obj"
        self.xsize = xsize
        self.zsize = zsize

        self.x_domain = BaseFunc.generate_discrete_points(xsize, xdim).float()
        self.z_domain = BaseFunc.generate_discrete_points(zsize, zdim).float()
        self.xz_domain = self.get_discrete_xz_domain()

    def get_discrete_x_domain(self):
        return self.x_domain

    def get_discrete_z_domain(self):
        return self.z_domain

    def get_beta_t(self, t):
        domain_size = self.xsize * self.zsize
        return 2.0 * np.log(domain_size * (t + 1) ** 2 / 6 / 0.1) / 100

    def get_noiseless_observation(self, xz):
        with torch.no_grad():
            xz = xz.reshape(-1, self.dim)
            xz = xz * torch.tensor(
                [50.0 - 20.0, 10.0 - 1.0, 50.0 - 20.0, 60.0 - 0.1]
            ) + torch.tensor([20.0, 1.0, 20.0, 0.1])

            val = (
                8.61
                * 1e5
                * torch.sqrt(xz[:, 0])
                * xz[:, 1]
                * torch.pow(xz[:, 2], -2.0 / 3.0)
                * torch.pow(xz[:, 3], -0.5)
                + 3.69 * 1e4 * xz[:, 2]
                + 7.72 * 1e8 / xz[:, 0] * torch.pow(xz[:, 1], 0.219)
                - 765.43 * 1e6 / xz[:, 0]
            ) / 1e4

            val = (val - 173.9976) / (3553.9161 - 173.9976)
            val = (val - 0.5) * 2.0

            val = val.reshape(
                -1,
            )
        return val.float()

    def get_z_domain_probability(self):
        if self.z_probabilities is not None:
            return self.z_probabilities

        with torch.no_grad():
            mu = self.params["z_distribution"]["mu"]
            sigma = self.params["z_distribution"]["sigma"]
            probabilities = (
                1.0
                / np.sqrt(2.0 * np.pi)
                / sigma
                * torch.exp(-0.5 * (self.z_domain - mu) ** 2 / sigma**2)
            ).squeeze()

            self.z_probabilities = probabilities / torch.sum(probabilities)
        return self.z_probabilities
