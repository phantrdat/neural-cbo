import os
import numpy as np
import torch
import math
from .basefunc import BaseFunc


class SpeedReducer(BaseFunc):
	def __init__(
		self,
		xsize=100,
		zsize=100,
		transformation="",
		noise_std=0.01,
	):
		xdim = 4
		zdim = 3

		super(SpeedReducer, self).__init__(
			xdim, zdim, xsize, zsize, transformation, noise_std=noise_std
		)

		self.module_name = "speed_reducer"
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
			
			
			lbound = torch.tensor([2.6, 0.7, 17.0, 7.3, 7.8, 2.9, 5.0])
			ubound = torch.tensor([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5])

			xz = xz * (ubound - lbound) + lbound
			
			x1, x2, x3, x4, x5, x6, x7 = xz.unbind(-1)
			val = - (
			0.7854 * x1 * x2.pow(2) * (3.3333 * x3.pow(2) + 14.9334 * x3 - 43.0934)
			+ -1.508 * x1 * (x6.pow(2) + x7.pow(2))
			+ 7.4777 * (x6.pow(3) + x7.pow(3))
			+ 0.7854 * (x4 * x6.pow(2) + x5 * x7.pow(2)) 
			).reshape(
				-1,
			)

			
		return val
	