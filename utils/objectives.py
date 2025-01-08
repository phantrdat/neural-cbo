from abc import abstractmethod
from stringprep import c22_specials
import numpy as np
import torch

from torch.distributions.uniform  import Uniform
from numpy.core.records import array
import torch
from torch.quasirandom import SobolEngine
import math, os
from torchvision.datasets import MNIST
import torch.nn as nn
import cv2
from torchmetrics.image import StructuralSimilarityIndexMeasure
class Objective():
	def __init__(self, dim, **kwargs) -> None:
		self.dim = dim
		self.min = torch.full([dim],0)
		self.max = torch.full([dim],1)

		if 'noise_std' in kwargs:
			self.noise_stds = kwargs['noise_std']
		else:
			self.noise_stds = None
		self.features = None
		self.target_observations = None
	def noise_std_estimate(self):
		points = self.generate_features(10000)
		
		objective_noise = [self.value(x, is_noise=False) for x in points]
		constraint_noises= [self.constraints(x, is_noise=False) for x in points]

		# variance of noise equals to 1 percent of function range
		objective_noise = 0.05*torch.std(torch.FloatTensor(objective_noise))
		constraint_noises = 0.05*torch.std(torch.FloatTensor(constraint_noises), dim=0)
		
		noise_stds = torch.cat((objective_noise.unsqueeze(0), constraint_noises))
		return noise_stds
	
	def generate_features(self, sample_num, seed=1504):
		# self.features = torch.FloatTensor(sample_num,self.dim).uniform_(self.min ,self.max)
		# self.features = torch.rand(sample_num,self.dim).to(self.min.device)

		sobol = SobolEngine(dimension=self.dim, scramble=True, seed=seed)
		self.features = sobol.draw(n=sample_num).to(device=self.min.device)
		self.features = torch.mul(self.max-self.min, self.features) + self.min
		return self.features.double()

	def generate_samples(self, sample_num, seed=1504, is_noise=True):
		self.generate_features(sample_num, seed=seed)
		self.target_observations = []
		self.constraint_observations = []

		for x in self.features:
			self.target_observations.append(self.value(x, is_noise=is_noise))
			self.constraint_observations.append(self.constraints(x, is_noise=is_noise))
		self.target_observations = torch.DoubleTensor(self.target_observations)
		self.constraint_observations =  torch.DoubleTensor(self.constraint_observations)
		points = {'features': self.features, 'observations':self.target_observations, 
			'constraint_observations': self.constraint_observations.T}
		return points			

	@abstractmethod  
	def value(self, X, is_noise):
		pass
	@abstractmethod
	def constraints(self,X, is_noise):
		pass

	
	@property
	def func_name(self):
		pass

class Gardner(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = torch.full([dim],0)
		self.max = torch.full([dim], 6)
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = torch.normal(mean=0.0, std=self.noise_stds[0])
		else:
			f_noise = 0
		target = torch.sin(X[0]) + X[1] + f_noise
		return target
	def constraints(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [torch.normal(mean=0.0, std=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0]
		
		c0 = torch.prod(torch.sin(X), dim=0) + 0.95 + c_noises[0] 
		return [c0]
	@property
	def func_name(self):
		return "Gardner"


class BraninHoo(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = torch.Tensor([-5, 0])
		self.max  = torch.Tensor([10, 15])
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()	
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = torch.normal(mean=0.0, std=self.noise_stds[0])
		else:
			f_noise = 0

		a = 1
		b = 5.1/(4*torch.pi*torch.pi)
		c = 5/torch.pi
		r = 6
		s = 10
		t = 1/(8*torch.pi)
		
		term1 = (X[1]-b*X[0]*X[0]+c*X[0]-r)
		term1 = a*term1**2
		term2 = s*(1-t)*torch.cos(X[0]) + s
		target  = term1 + term2 + f_noise
		return target
	def constraints(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [torch.normal(mean=0.0, std=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0]
		
		c0 = (X[0]-2.5)**2 + (X[1]-7.5)**2 - 50 + c_noises[0]
		return [c0]
	@property
	def func_name(self):
		return "BraninHoo"
	
class Gramacy(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = torch.Tensor([0, 0])
		self.max  = torch.Tensor([1, 1])
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()	
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = torch.normal(mean=0.0, std=self.noise_stds[0])
		else:
			f_noise = 0
		# target = (X[0]-0.8)**2 + (X[1]-0.7)**2 + f_noise
		target = X[0] + X[1] + f_noise
		return target
	def constraints(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [torch.normal(mean=0.0, std=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0, 0]
		
		c0 = - 0.5*torch.sin(2*torch.pi*(X[0]**2 - 2*X[1])) - X[0] - 2*X[1] + 1.5 + c_noises[0]
		c1 = X[0]**2 + X[1]**2 - 1.5 + c_noises[1]
		return  [c0, c1]

	@property
	def func_name(self):
		return "Gramacy"
	
class Simionescu(Objective):
	def __init__(self, dim=2) -> None:
		super().__init__(dim)
		self.min = torch.Tensor([-1.25, -1.25])
		self.max  = torch.Tensor([1.25, 1.25])
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()	
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = torch.normal(mean=0.0, std=self.noise_stds[0])
		else:
			f_noise = 0
		target = 0.1*X[0]*X[1] + f_noise
		return target
	def constraints(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [torch.normal(mean=0.0, std=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0, 0]
		rT = 1
		rS = 0.2 
		n = 8
		c0 = X[0]**2 + X[1]**2 -(rT + rS*torch.cos(n*torch.arctan (X[0]/X[1])))**2  + c_noises[0]
		return  [c0]

	@property
	def func_name(self):
		return "Simionescu"
	
class Gomez_and_Levy(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = torch.Tensor([-1, -1])
		self.max  = torch.Tensor([0.75, 1])
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()	
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = torch.normal(mean=0.0, std=self.noise_stds[0])
		else:
			f_noise = 0
		target = 4*X[0]**2 - 2.1*X[0]**4 + (X[0]**6)/3 + X[0]*X[1] - 4*X[1]**2 + 4*X[1]**4 + f_noise
		return target
	def constraints(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [torch.normal(mean=0.0, std=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0]
		c0 = -torch.sin(4*torch.pi*X[0]) + 2*torch.sin(2*torch.pi*X[1])**2 - 1.5 + c_noises[0]
		return  [c0]

	@property
	def func_name(self):
		return "Gomez_and_Levy"



class Hartmann(Objective):
	def __init__(self, dim=6) -> None:
		super().__init__(dim)
		self.min = torch.zeros(dim)
		self.max  = torch.ones(dim)
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()	
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = torch.normal(mean=0.0, std=self.noise_stds[0])
		else:
			f_noise = 0
		device = X.device

		A = torch.tensor([[10, 3, 17, 3.5, 1.7, 8],
			[0.05, 10, 17, 0.1, 8, 14],
			[3, 3.5, 1.7, 10, 17, 8],
			[17, 8, 0.05, 10, 0.1, 14]]).to(device)
		P = 0.0001*torch.tensor([[1312, 1696, 5569, 124, 8283, 5886],
				[2329, 4135, 8307, 3736, 1004, 9991],
				[2348, 1451, 3522, 2883, 3047, 6650],
				[4047, 8828, 8732, 5743, 1091, 381.0],
			]).to(device)

		alpha = torch.tensor([1.0, 1.2, 3.0, 3.2]).to(device)
		
		inner_sum = torch.sum(
			A * (X.unsqueeze(-2) -  P).pow(2), dim=-1
		)
		target = -(torch.sum(alpha * torch.exp(-inner_sum), dim=-1)) + f_noise
		return target
	def constraints(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [torch.normal(mean=0.0, std=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0]
		c0 = X.norm() - 1 + c_noises[0]
		return  [c0]

	@property
	def func_name(self):
		return "Hartmann"

class Ackley(Objective):
	def __init__(self, dim=5) -> None:
		super().__init__(dim)
		self.min = torch.tensor([-5]*dim)
		self.max  = torch.tensor([3]*dim)
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()	
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = torch.normal(mean=0.0, std=self.noise_stds[0])
		else:
			f_noise = 0
			
		a = 20
		b = .2
		c = 2 * math.pi
		part1 = - a * torch.exp(-b / math.sqrt(self.dim) * torch.linalg.norm(X, dim=-1))
		part2 = - (torch.exp(torch.mean(torch.cos(c * X), dim=-1)))
		target =  part1 + part2 + a + math.e + f_noise 
		return target
	def constraints(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [torch.normal(mean=0.0, std=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0, 0]
		c0 = 1 - (torch.linalg.norm(X-1, ord=2, dim=-1) - 5.5)**2 + c_noises[0]
		c1 = torch.linalg.norm(X, ord=torch.inf, dim=-1)**2  - 9  + c_noises[1]
		return  [c0, c1]
	@property
	def func_name(self):
		return "Ackley"






class WeldedBeamSO(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = torch.Tensor([0.125, 0.1, 0.1, 0.1])
		self.max  = torch.Tensor([10, 10, 10, 10])
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()	
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = torch.normal(mean=0.0, std=self.noise_stds[0])
		else:
			f_noise = 0
		
		target = 1.10471*X[0]**2*X[1] + 0.04811*X[2]*X[3]*(14+X[1])+ f_noise
		return target
	def constraints(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [torch.normal(mean=0.0, std=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0,0,0,0,0,0]

		x1, x2, x3, x4 = X.unbind(-1)
		P = 6000.0
		L = 14.0
		E = 30e6
		G = 12e6
		t_max = 13600.0
		s_max = 30000.0
		d_max = 0.25

		M = P * (L + x2 / 2)
		R = torch.sqrt(0.25 * (x2.pow(2) + (x1 + x3).pow(2)))
		J = 2 * math.sqrt(2) * x1 * x2 * (x2.pow(2) / 12 + 0.25 * (x1 + x3).pow(2))
		P_c = (
			4.013
			* E
			* x3
			* x4.pow(3)
			* 6
			/ (L**2)
			* (1 - 0.25 * x3 * math.sqrt(E / G) / L)
		)
		t1 = P / (math.sqrt(2) * x1 * x2)
		t2 = M * R / J
		t = torch.sqrt(t1.pow(2) + t1 * t2 * x2 / R + t2.pow(2))
		s = 6 * P * L / (x4 * x3.pow(2))
		d = 4 * P * L**3 / (E * x3.pow(3) * x4)

		c0 = t - t_max 
		c1 = s - s_max 
		c2 = x1 - x4 
		c3 = 0.10471 * x1.pow(2) + 0.04811 * x3 * x4 * (14.0 + x2) - 5.0 
		c4 = d - d_max 
		c5 = P - P_c
		
		return  [c0,c1,c2,c3,c4,c5]

	@property
	def func_name(self):
		return "WeldedBeamSO"


class SpeedReducer(Objective):
	def __init__(self, dim=7) -> None:
		super().__init__(dim)
		self.min = torch.tensor([2.6, 0.7, 17.0, 7.3, 7.8, 2.9, 5.0])
		self.max = torch.tensor([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5])
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()	
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = torch.normal(mean=0.0, std=self.noise_stds[0])
		else:
			f_noise = 0
			
		x1, x2, x3, x4, x5, x6, x7 = X.unbind(-1)
		return (
			0.7854 * x1 * x2.pow(2) * (3.3333 * x3.pow(2) + 14.9334 * x3 - 43.0934)
			+ -1.508 * x1 * (x6.pow(2) + x7.pow(2))
			+ 7.4777 * (x6.pow(3) + x7.pow(3))
			+ 0.7854 * (x4 * x6.pow(2) + x5 * x7.pow(2)) 
		) + f_noise
	def constraints(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [torch.normal(mean=0.0, std=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0]*11
		x1, x2, x3, x4, x5, x6, x7 = X.unbind(-1)
		c0 = 27.0 * (1 / x1) * (1 / x2.pow(2)) * (1 / x3) - 1  + c_noises[0]
		c1 = 397.5 * (1 / x1) * (1 / x2.pow(2)) * (1 / x3.pow(2)) - 1 + c_noises[1]
		c2 = 1.93 * (1 / x2) * (1 / x3) * x4.pow(3) * (1 / x6.pow(4)) - 1 + c_noises[2]
		c3 = 1.93 * (1 / x2) * (1 / x3) * x5.pow(3) * (1 / x7.pow(4)) - 1 + c_noises[3]
		c4 = 1/ (0.1 * x6.pow(3))* torch.sqrt((745 * x4 / (x2 * x3)).pow(2) + 16.9 * 1e6) - 1100 + c_noises[4]
		c5 = 1/ (0.1 * x7.pow(3))* torch.sqrt((745 * x5 / (x2 * x3)).pow(2) + 157.5 * 1e6)- 850 + c_noises[5]
		c6 = x2 * x3 - 40 + c_noises[6]
		c7 = 5 - x1 / x2 + c_noises[7]
		c8 = x1 / x2 - 12 + c_noises[8]
		c9 = (1.5 * x6 + 1.9) / x4 - 1 + c_noises[9]
		c10 = (1.1 * x7 + 1.9) / x5 - 1 + c_noises[10]
		
		return  [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]

	@property
	def func_name(self):
		return "SpeedReducer"

class GasTransmission(Objective):
	def __init__(self, dim=4) -> None:
		super().__init__(dim)	
		self.min = torch.tensor([20., 1., 20., 0.1])
		self.max = torch.tensor([50., 10., 50., 60.])
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()	
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = torch.normal(mean=0.0, std=self.noise_stds[0])
		else:
			f_noise = 0
			
		x1, x2, x3, x4 = X.unbind(-1)
		
		f =  (
			8.61 * 10 * torch.sqrt(x1) * x2 * x3.pow(-2./3.) * x4.pow(-0.5) 
			+ 3.69 * x3 
			+ 7.72 * 1e4 * x1.pow(-1) * x2.pow(0.219) 
			- 765.43 * 1e2 * x1.pow(-1)  
		)
		
		f = (f - 173.9976) / (3553.9161 - 173.9976)
		f = (f - 0.5) * 2.0
		
		return  -f + f_noise
	
	def constraints(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [torch.normal(mean=0.0, std=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0]
		x1, x2, x3, x4 = X.unbind(-1)
		
		c0 = x4/x2.pow(2) + x2.pow(2) - 1
		 
		max_val = 99.3931
		min_val = 0.4373
		
		c0 = (c0 - min_val) / (max_val - min_val)
		c0 = (c0 - 0.5) * 2.0 + 0.5 + c_noises[0]
		return  [c0]

	@property
	def func_name(self):
		return "GasTransmission"





# # Real world tasks


# # ** Sensitive samples 
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
	

class SensitiveSample(Objective):
	def __init__(self, dim=49, trained_model_path='models_and_data/pretrained_models/trained_MNIST_acc_0.93.pth', gpu_id=0) -> None:
		self.min= torch.full([dim],-0.42421)
		self.max= torch.full([dim], 2.8215)
		self.dim=dim
		self.device = torch.device(gpu_id if torch.cuda.is_available() else 'cpu')
		self.dataset = MNIST_Sensitive_Loader(root='models_and_data/mnist_data/', device=self.device, train=True, download=True)
		random_v0_idx = np.random.randint(low=0, high=50000)
		# print("random v0:", random_v0_idx)
		self.v0 , _ =  self.dataset.__getitem__(random_v0_idx)
		if os.path.isfile(trained_model_path):
			self.model = MLP_MNIST(dropout=0.0, num_classes=10, n_hidden=10).to(self.device)
			self.model.load_state_dict(torch.load(trained_model_path))
			self.model.eval()
		else:
			print("Trained model is not existed")
		
	def value(self,X,is_noise=False):
		def grad_norm(X):
			X = (0.3081*X+0.1307)*255
			X = torch.FloatTensor(cv2.resize(X.cpu().numpy(), (28,28))).to(self.device)
			X = X.div(255)
			X = X.sub_(0.1307).div_(0.3081)
			y = self.model(X)
			y = torch.max(y)
			self.model.zero_grad()
			y.backward()
			grad = torch.cat(
					[w.grad.detach().flatten() for w in self.model.parameters() if w.requires_grad]
			).to(self.device)
			return torch.norm(grad, p='fro')
		return -grad_norm(X)
	def generate_features(self, sample_num, seed=1504):
		self.features = []
		max_pertubed = torch.full([self.dim], 0.7).to(self.device)
		min_pertubed = torch.full([self.dim], -0.2).to(self.device)
		# offset = super().generate_features(sample_num, seed=seed).to(self.device)
		sobol = SobolEngine(dimension=self.dim, scramble=True, seed=seed)
		offset = sobol.draw(n=sample_num).to(self.device)
		offset = torch.mul(max_pertubed-min_pertubed, offset) + min_pertubed
	
		self.features = torch.stack([torch.clamp(self.v0.flatten() + x, min=-0.42421, max=2.8215) for x in offset]).double()
		# indexes = np.random.randint(0, len(self.dataset),size=sample_num)
		# self.features, _ = self.dataset.__getitem__(indexes)
		# for i in range(sample_num):
		# 	index = np.random.randint(0, len(dataset))
		# 	sample,_  = dataset.__getitem__(index)
		# 	sample = sample.flatten().numpy()
		# 	self.features.append(sample)
		# self.features = torch.flatten(self.features, start_dim=1)
		return self.features
	def constraints(self, X, is_noise=True):
		# sz = int(np.sqrt(self.dim))
		# ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
		# score = ssim(X.resize_(sz, sz).unsqueeze(0).unsqueeze(0).cpu(), self.v0.unsqueeze(0).cpu())
		score = nn.CosineSimilarity(dim=0)(X, self.v0.flatten())
		return [0.9 - score]
		
	@property
	def func_name(self):
		return "SensitiveSample"







if __name__ == "__main__":
	
	# x1, x2, x3, x4 = [0.9592, 0.0176, 0.3632, 0.3849]
	
	# x1 = x1 * (50.0 - 20.0) + 20
	# x2 = x2 * (10.0 - 1.0) + 1
	# x3 = x3 * (50.0 - 20.0) + 20
	# x4 = x4 * (60.0 - 0.1) + 0.1
	# X = torch.Tensor([x1, x2, x3, x4])
	best_features = []
	for run in range(10):
		problem = SensitiveSample(dim=49)
		features = problem.generate_features(10000, seed=0).to("cuda:0")
		values = torch.tensor([problem.value(x, is_noise=False) for x in features])
		
		constraints = torch.tensor([problem.constraints(x, is_noise=False)  for x in features])
		fea_idx = torch.where(constraints <= 0)[0]
		values = values[fea_idx]
		constraints = constraints[fea_idx]
		features = features[fea_idx]
		
		fea_min_idx = torch.argmin(values)
		best_features.append(features[fea_min_idx])
		# print(values[fea_min_idx], features[fea_min_idx])
	print(best_features)
# tensor([-1.1991, -0.2598, -1.0442, -2.4892, -4.9679])
	
	# X = features[fea_min_idx]
	
	# print(problem.value(X, is_noise=False).item(), problem.constraints(X, is_noise=False))
	
	# X = torch.tensor([[20.2302,  5.0754, 20.8700,  0.1013]])
	
	# print(problem.value(X, is_noise=False).item(), problem.constraints(X, is_noise=False))