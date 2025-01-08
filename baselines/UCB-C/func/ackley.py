import os
import numpy as np
import torch
import math
from .basefunc import BaseFunc


class Ackley(BaseFunc):
    def __init__(
        self,
        xsize=100,
        zsize=100,
        transformation="",
        noise_std=0.01,
    ):
        xdim = 4
        zdim = 1

        super(Ackley, self).__init__(
            xdim, zdim, xsize, zsize, transformation, noise_std=noise_std
        )

        self.module_name = "ackley"
        self.xsize = xsize
        self.zsize = zsize

        self.x_domain = BaseFunc.generate_discrete_points(xsize, xdim, low=-5, high=3)
        self.z_domain = BaseFunc.generate_discrete_points(zsize, zdim, low=-5, high=3)
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

            a = 20
            b = .2
            c = 2 * math.pi
            part1 = - a * torch.exp(-b / math.sqrt(self.dim) * torch.linalg.norm(xz, dim=-1))
            part2 = - (torch.exp(torch.mean(torch.cos(c * xz), dim=-1)))
            val =  - (part1 + part2 + a + math.e)
            val = val.reshape(
                -1,
            )
        return val
