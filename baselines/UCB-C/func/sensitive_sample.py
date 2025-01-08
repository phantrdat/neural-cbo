import os
import numpy as np
import torch
import math
from .basefunc import BaseFunc
from torchvision.datasets import MNIST
import torch.nn as nn
import cv2
class MLP_MNIST(nn.Module):
	def __init__(self, dropout = 0.2, num_classes=10, n_hidden = 100):
		super(MLP_MNIST, self).__init__()
		self.layer1 = nn.Linear(in_features=784, out_features=n_hidden)
		self.layer2 = nn.Linear(in_features=n_hidden, out_features=num_classes)
		self.activation = nn.ReLU()
		self.log_softmax = nn.LogSoftmax(dim=1)
	def forward(self, x):
		x = x.view(-1, 784)
		out = self.layer1(x)
		out = self.activation(out)
		out = self.layer2(out)
		out = self.log_softmax(out)
		return out

class MNIST_Sensitive_Loader(MNIST):
	def __init__(self, device, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.data = torch.FloatTensor([cv2.resize(x.numpy(), (7,7)) for x in self.data])
		# Scale data to [0,1]
		self.data = self.data.unsqueeze(1).float().div(255)

		# Normalize it with the usual MNIST mean and std
		self.data = self.data.sub_(0.1307).div_(0.3081)

		# Put both data and targets on GPU in advance
		self.data, self.targets = self.data.to(device), self.targets.to(device)

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target = self.data[index], self.targets[index]

		return img, target
	
class SensitiveSample(BaseFunc):
	def __init__(
		self,
		xsize=100,
		zsize=100,
		transformation="",
		noise_std=0.01,
	):
		xdim = 24
		zdim = 25

		super(SensitiveSample, self).__init__(
			xdim, zdim, xsize, zsize, transformation, noise_std=noise_std
		)

		self.module_name = "sensitive_sample"
		self.xsize = xsize
		self.zsize = zsize

		self.device = torch.device('cpu')
		self.dataset = MNIST_Sensitive_Loader(root='/home/trongp/constrained_NeuralBO/models_and_data/mnist_data/', device=self.device, train=True, download=True)
		self.v0 , _ =  self.dataset.__getitem__(1000)
		
		self.x_domain = BaseFunc.generate_discrete_points(xsize, xdim, low=-0.2, high=0.7) 
		self.z_domain = BaseFunc.generate_discrete_points(zsize, zdim, low=-0.2, high=0.7) 
		self.xz_domain = self.get_discrete_xz_domain()  + self.v0.flatten()
		self.xz_domain = np.clip(self.get_discrete_xz_domain() + self.v0.flatten(), a_min=-0.42421, a_max=2.8215)
		
		trained_model_path='/home/trongp/constrained_NeuralBO/models_and_data/pretrained_models/trained_MNIST_acc_0.93.pth'
		
		
		
		
		if os.path.isfile(trained_model_path):
			self.model = MLP_MNIST(dropout=0.0, num_classes=10, n_hidden=10).to(self.device)
			self.model.load_state_dict(torch.load(trained_model_path))
			self.model.eval()
		else:
			print("Trained model is not existed")

	def get_discrete_x_domain(self):
		return self.x_domain

	def get_discrete_z_domain(self):
		return self.z_domain

	def get_beta_t(self, t):
		domain_size = self.xsize * self.zsize
		return 2.0 * np.log(domain_size * (t + 1) ** 2 / 6 / 0.1) / 100
	
	def get_noiseless_observation(self, xz):
		def grad_norm(X):
			# X.requires_grad = True
		
			grad_norms = []
			for x_i in X:
				x_i = (0.3081*x_i+0.1307)*255
				x_i = torch.FloatTensor(cv2.resize(x_i.cpu().numpy(), (28,28))).to(self.device)
				x_i = x_i.div(255)
				x_i = x_i.sub_(0.1307).div_(0.3081)
				y = self.model(x_i)
				y = torch.max(y)
				self.model.zero_grad()
				y.backward()
				grad = torch.cat(
						[w.grad.detach().flatten() for w in self.model.parameters() if w.requires_grad]
				).to(self.device)
				grad_norms.append(torch.norm(grad, p='fro'))
			grad_norms = torch.stack(grad_norms)
			return grad_norms
		# with torch.no_grad():
		xz_prime = xz.reshape(-1, self.dim)
		val = grad_norm(xz_prime)
		return val
