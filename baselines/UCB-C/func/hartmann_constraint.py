import os
import numpy as np
import torch


from .basefunc import BaseFunc


class Hartmann_Constraint(BaseFunc):
    def __init__(
        self,
        xsize=10,
        zsize=10,
        transformation="",
        noise_std=0.01,
    ):
        xdim = 3
        zdim = 3

        super(Hartmann_Constraint, self).__init__(
            xdim, 
            zdim, 
            xsize, 
            zsize, 
            transformation, 
            noise_std=noise_std
        )

        self.module_name = "hartmann_constraint"
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
        return 2.0 * np.log(domain_size * (t + 1) ** 2 / 6 / 0.1) / 20

    def get_noiseless_observation(self, xz):
        with torch.no_grad():
            xz = xz.reshape(-1, self.dim)
            
            val = xz.norm(dim=-1) - 1
            
            val = val.reshape(
                -1,
            )
        return val
