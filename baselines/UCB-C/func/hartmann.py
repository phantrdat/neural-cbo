import os
import numpy as np
import torch

from .basefunc import BaseFunc


class Hartmann(BaseFunc):
    def __init__(
        self,
        xsize=100,
        zsize=100,
        transformation="",
        noise_std=0.01,
    ):
        xdim = 3
        zdim = 3

        super(Hartmann, self).__init__(
            xdim, zdim, xsize, zsize, transformation, noise_std=noise_std
        )

        self.module_name = "hartmann"
        self.xsize = xsize
        self.zsize = zsize

        self.x_domain = BaseFunc.generate_discrete_points(xsize, xdim, low=0, high=1)
        self.z_domain = BaseFunc.generate_discrete_points(zsize, zdim, low=0, high=1)
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

            A = torch.tensor([[10, 3, 17, 3.5, 1.7, 8],
			[0.05, 10, 17, 0.1, 8, 14],
			[3, 3.5, 1.7, 10, 17, 8],
			[17, 8, 0.05, 10, 0.1, 14]])
            
            P = 0.0001*torch.tensor([[1312, 1696, 5569, 124, 8283, 5886],
                    [2329, 4135, 8307, 3736, 1004, 9991],
                    [2348, 1451, 3522, 2883, 3047, 6650],
                    [4047, 8828, 8732, 5743, 1091, 381.0],
                ])
            
            alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])
            
            inner_sum = torch.sum(
                A * (xz.unsqueeze(-2) -  P).pow(2), dim=-1
            )
            
            val = (torch.sum(alpha * torch.exp(-inner_sum), dim=-1))
            val = val.reshape(
                -1,
            )
        return val
