import os
import numpy as np
import torch
import json
from torch.quasirandom import SobolEngine
class BaseFunc:
    def __init__(self, xdim, zdim, xsize, zsize, transformation="", noise_std=None):
        self.xdim = xdim
        self.zdim = zdim
        self.xsize = xsize
        self.zsize = zsize
        self.dim = self.xdim + self.zdim
        self.transformation = transformation
        self.noise_std = noise_std

        self.xz_domain = None
        self.maximizer = None
        self.maximum_value = None

    @staticmethod
    def generate_discrete_points(n, dim=1, low=0.0, high=1.0):
        if dim == 1:
            return torch.linspace(low, high, n).reshape(-1, 1)
        elif dim > 1:
            # rand01 = np.loadtxt("/home/trongp/constrained_NeuralBO/baselines/UCB-C/initial_inputs.txt")
            # return torch.from_numpy(rand01[: n * dim]).reshape(n, dim)
            sobol = SobolEngine(dimension=dim, scramble=True, seed=0)
            points = sobol.draw(n)
            points = torch.mul(high-low, points) + low
            return points
        else:
            raise Exception("Dimension must be positive!")

    def get_GP_hyperparameters(self, transformation):
        hyperparameter_filename = f"optimization_results/hyperparameters/{self.module_name}__{transformation}.json"

        with open(hyperparameter_filename, "r") as f:
            hyperparameters = json.load(f)
        return hyperparameters

    def get_params(self, transformation):
        raise Exception("To be implemented in child class!")

    def get_discrete_x_domain(self):
        raise Exception("To be implemented in child class!")

    def get_discrete_z_domain(self):
        raise Exception("To be implemented in child class!")

    def get_discrete_xz_domain(self):
        if self.xz_domain is not None:
            return self.xz_domain

        with torch.no_grad():
            x_domain = self.get_discrete_x_domain()
            z_domain = self.get_discrete_z_domain()
            repeat_interleave_x_domain = x_domain.repeat_interleave(
                z_domain.shape[0], dim=0
            )
            repeat_z_domain = z_domain.repeat(x_domain.shape[0], 1)
            self.xz_domain = torch.concat(
                [repeat_interleave_x_domain, repeat_z_domain],
                dim=1,
            )
        return self.xz_domain

    def get_init_observations(self, n, seed=0):
        torch.manual_seed(seed)

        # with torch.no_grad():
        init_idxs = torch.randint(low=0, high=self.xz_domain.shape[0], size=(n,))
        init_xz = self.xz_domain[init_idxs]
        init_y = self.get_noisy_observation(init_xz)

        return init_xz, init_y

    def get_maximizer(self):
        if self.maximizer is not None:
            return self.maximizer

        # with torch.no_grad():
        func_range = self.get_noiseless_observation(self.xz_domain)
        max_idx = torch.argmax(func_range)

        self.maximizer = self.xz_domain[max_idx]
        self.maximum_value = func_range[max_idx].squeeze()

        return self.maximizer

    def get_maximum(self):
        if self.maximum_value is None:
            self.get_maximizer()
        return self.maximum_value

    def get_max_func_eval(self):
        # with torch.no_grad():
        func_evals = self.get_noiseless_observation(self.xz_domain)
        return torch.max(func_evals)

    def get_max_constrained_func_eval(self, constraint_func, threshold):
        # with torch.no_grad():
        constraint_range = constraint_func.get_noiseless_observation(self.xz_domain)
        penalty = -(constraint_range < threshold) * 1e9
        constrained_func_evals = (
                self.get_noiseless_observation(self.xz_domain) + penalty
            )
        return torch.max(constrained_func_evals)

    def get_noisy_observation(self, xz):
        if self.noise_std is None:
            raise Exception("Unknown noise")
        # with torch.no_grad():
        xz = xz.reshape(-1, self.dim)
        n_obs = xz.shape[0]
        return self.get_noiseless_observation(xz) + torch.randn(n_obs) * self.noise_std
