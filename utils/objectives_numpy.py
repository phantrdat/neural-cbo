from abc import abstractmethod
from stringprep import c22_specials
import numpy as np
from numpy.core.records import array
from torch.quasirandom import SobolEngine
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.datasets import MNIST
import torch.nn as nn
import os
import cv2
class Objective():
	def __init__(self, dim, **kwargs) -> None:
		self.dim = dim
		self.min = np.full([dim],0)
		self.max = np.full([dim],1)

		if 'noise_std' in kwargs:
			self.noise_stds = kwargs['noise_std']
		else:
			self.noise_stds = None
		self.features = None
		self.target_observations = None
	def noise_std_estimate(self):
		points = self.generate_features(10000)
		
		objective_noise = self.value(points, is_noise=False)
		constraint_noises= []
		for c in self.constraints(is_noise=False):
			constraint_noises.append(c(points))
		constraint_noises = np.stack(constraint_noises)

		# variance of noise equals to 1 percent of function range
		objective_noise = 0.05*np.std(objective_noise)
		constraint_noises = 0.05*np.std(constraint_noises, axis=1)
		
		noise_stds = np.concatenate((np.expand_dims(objective_noise, 0), constraint_noises))
		return noise_stds
	
	def generate_features(self, sample_num, seed=1504):
		sobol = SobolEngine(dimension=self.dim, scramble=True, seed=seed)
		self.features = sobol.draw(n=sample_num).numpy()
		self.features = np.multiply(self.max-self.min, self.features) + self.min
		return np.array(self.features, dtype=np.float64)

	# def generate_samples(self, sample_num, seed=1504):
	# 	self.generate_features(sample_num, seed=seed)
	# 	self.target_observations = []
	# 	self.constraint_observations = []

	# 	for x in self.features:
	# 		self.target_observations.append(self.value(x))
	# 		self.constraint_observations.append(self.constraints(x))
	# 	self.target_observations = np.array(self.target_observations, dtype=np.float64)
	# 	self.constraint_observations =  np.array(self.constraint_observations, dtype=np.float64)
	# 	points = {'features': self.features, 'observations':self.target_observations, 
	# 		'constraint_observations': self.constraint_observations.T}
	# 	return points			

	@abstractmethod  
	def value(self, X, is_noise):
		pass
	@abstractmethod
	def constraints(self, is_noise):
		pass

	
	@property
	def func_name(self):
		pass

class Gardner(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = np.full([dim], 0)
		self.max = np.full([dim], 6)
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = np.random.normal(loc=0.0, scale=self.noise_stds[0])
		else:
			f_noise = 0
		# target = np.sin(X[:,0]) + X[:,1]
		target = np.sin(X[:,0]) + X[:,1] + f_noise
		return target
	def constraints(self, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [np.random.normal(loc=0.0, scale=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0]
		
		c0 = lambda X: np.prod(np.sin(X), axis=1) + 0.95 + c_noises[0]
		return [c0]
	@property
	def func_name(self):
		return "Gardner"

class Gramacy(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = np.full([dim], 0)
		self.max  = np.full([dim], 1)
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()	
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = np.random.normal(loc=0.0, scale=self.noise_stds[0])
		else:
			f_noise = 0
		target = X[:,0] + X[:,1] + f_noise
		return target
	def constraints(self, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [np.random.normal(loc=0.0, scale=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0, 0]
		
		# def c0(X):
		# 	return - 0.5*np.sin(2*np.pi*(X[:,0]**2 - 2*X[:,1])) - X[:,0] - 2*X[:,1] + 1.5 + c_noises[0]
		# def c1(X):
		# 	return  X[:,0]**2 + X[:,1]**2 - 1.5 + c_noises[1]
		c0 = lambda X: - 0.5*np.sin(2*np.pi*(X[:,0]**2 - 2*X[:,1])) - X[:,0] - 2*X[:,1] + 1.5 + c_noises[0]
		c1 = lambda X: X[:,0]**2 + X[:,1]**2 - 1.5 + c_noises[1]
		return  [c0, c1]

	@property
	def func_name(self):
		return "Gramacy"


class BraninHoo(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = np.array([-5, 0])
		self.max  = np.array([10, 15])
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()	
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = np.random.normal(loc=0.0, scale=self.noise_stds[0])
		else:
			f_noise = 0

		a = 1
		b = 5.1/(4*np.pi*np.pi)
		c = 5/np.pi
		r = 6
		s = 10
		t = 1/(8*np.pi)
		
		term1 = (X[:,1] - b*X[:,0]**2 + c*X[:,0]-r)
		term1 = a*term1**2
		term2 = s*(1 - t)*np.cos(X[:,0]) + s
		target  = term1 + term2 + f_noise
		return target
	def constraints(self, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [np.random.normal(loc=0.0, scale=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0]
		
		c0 = lambda X: (X[:,0]-2.5)**2 + (X[:,1]-7.5)**2 - 50 + c_noises[0]
		return [c0]
	@property
	def func_name(self):
		return "BraninHoo"
	

class Simionescu(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = np.array([-1.25, -1.25])
		self.max  = np.array([1.25, 1.25])
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()	
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = np.random.normal(loc=0.0, scale=self.noise_stds[0])
		else:
			f_noise = 0
		target = 0.1*X[:,0]*X[:,1] +  f_noise
		return target
	def constraints(self, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [np.random.normal(loc=0.0, scale=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0, 0]
		rT = 1
		rS = 0.2 
		n = 8
		c0 = lambda X: X[:,0]**2 + X[:,1]**2 -(rT + rS*np.cos(n*np.arctan (X[:,0]/X[:,1]) ))**2  + c_noises[0]
		return  [c0]

	@property
	def func_name(self):
		return "Simionescu"
	

class Gomez_and_Levy(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = np.array([-1, -1])
		self.max  = np.array([0.75, 1])
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()	
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = np.random.normal(loc=0.0, scale=self.noise_stds[0])
		else:
			f_noise = 0

		target = 4*X[:,0]**2 - 2.1*X[:,0]**4 + (X[:,0]**6)/3 + X[:,0]*X[:,1] - 4*X[:,1]**2 + 4*X[:,1]**4 + f_noise
		return target
	def constraints(self, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [np.random.normal(loc=0.0, scale=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0]
		
		c0 = lambda X: -np.sin(4*np.pi*X[:,0]) + 2*np.sin(2*np.pi*X[:,1])**2 - 1.5 + c_noises[0]
		return [c0]
	@property
	def func_name(self):
		return "Gomez_and_Levy"

class Hartmann(Objective):
	def __init__(self, dim=6) -> None:
		super().__init__(dim)
		self.min = np.zeros(dim)
		self.max  = np.ones(dim)
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()	
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = np.random.normal(loc=0.0, scale=self.noise_stds[0])
		else:
			f_noise = 0
		
		A = np.array([[10, 3, 17, 3.5, 1.7, 8],
			[0.05, 10, 17, 0.1, 8, 14],
			[3, 3.5, 1.7, 10, 17, 8],
			[17, 8, 0.05, 10, 0.1, 14]])
		P = 0.0001*np.array([[1312, 1696, 5569, 124, 8283, 5886],
				[2329, 4135, 8307, 3736, 1004, 9991],
				[2348, 1451, 3522, 2883, 3047, 6650],
				[4047, 8828, 8732, 5743, 1091, 381.0],
			])

		alpha = np.array([1.0, 1.2, 3.0, 3.2])
		inner_sum = np.sum(
            A * np.power((np.expand_dims(X, axis=-2) -  P), 2), axis=-1 )
		target = -(np.sum(alpha * np.exp(-inner_sum), axis=-1)) + f_noise
		return target
	def constraints(self, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [np.random.normal(loc=0.0, scale=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0]
		
		c0 = lambda X: np.linalg.norm(X, axis=-1) - 1.5 + c_noises[0]
		return [c0]
	@property
	def func_name(self):
		return "Hartmann"

class Ackley(Objective):
	def __init__(self, dim=5) -> None:
		super().__init__(dim)
		self.min = np.array([-5]*dim)
		self.max  = np.array([3]*dim)
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()	
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = np.random.normal(loc=0.0, scale=self.noise_stds[0])
		else:
			f_noise = 0
			
		a = 20
		b = .2
		c = 2 * np.pi
		part1 = -a * np.exp(-b / np.sqrt(self.dim) * np.linalg.norm(X, axis=-1))
		part2 = - (np.exp(np.mean(np.cos(c * X), axis=-1)))
		target =  part1 + part2 + a + np.e + f_noise 
		return target
	def constraints(self, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [np.random.normal(loc=0.0, scale=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0, 0]
		c0 = lambda X:  1 - (np.linalg.norm(X-1, ord=2, axis=-1) - 5.5)**2 + c_noises[0]	
		c1 = lambda X:  np.linalg.norm(X, ord=np.inf, axis=-1)**2 - 9  + c_noises[1]
		return  [c0, c1]
	@property
	def func_name(self):
		return "Ackley"

class SpeedReducer(Objective):
	def __init__(self, dim=7) -> None:
		super().__init__(dim)
		self.min = np.array([2.6, 0.7, 17.0, 7.3, 7.8, 2.9, 5.0])
		self.max = np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5])
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()	
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = np.random.normal(loc=0.0, scale=self.noise_stds[0])
		else:
			f_noise = 0
			
		x1, x2, x3, x4, x5, x6, x7 = X[:,0], X[:,1], X[:,2], X[:,3], X[:,4], X[:,5], X[:,6]
		return (
			0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
			+ -1.508 * x1 * (x6**2 + x7**2)
			+ 7.4777 * (x6**3 + x7**3)
			+ 0.7854 * (x4 * x6**2 + x5 * x7**2) 
		) + f_noise
	def constraints(self, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [np.random.normal(loc=0.0, scale=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0]*11
		
		
		c0 = lambda X: 27.0 * (1 / X[:,0]) * (1 / X[:,1]**2) * (1 / X[:,2]) - 1  + c_noises[0]
		c1 = lambda X: 397.5 * (1 / X[:,0]) * (1 / X[:,1]**2) * (1 / X[:,2]**2) - 1 + c_noises[1]
		c2 = lambda X: 1.93 * (1 / X[:,1]) * (1 / X[:,2]) * X[:,3]**3 * (1 / X[:,5]**4) - 1 + c_noises[2]
		c3 = lambda X: 1.93 * (1 / X[:,1]) * (1 / X[:,2]) * X[:,4]**3 * (1 / X[:,6]**4) - 1 + c_noises[3]
		c4 = lambda X: 1/ (0.1 * X[:,5]**3)* np.sqrt((745 * X[:,3] / (X[:,1] * X[:,2]))**2 + 16.9 * 1e6) - 1100 + c_noises[4]
		c5 = lambda X: 1/ (0.1 * X[:,6]**3)* np.sqrt((745 * X[:,4] / (X[:,1] * X[:,2]))**2 + 157.5 * 1e6)- 850 + c_noises[5]
		c6 = lambda X: X[:,1] * X[:,2] - 40 + c_noises[6]
		c7 = lambda X: 5 - X[:,0] / X[:,1] + c_noises[7]
		c8 = lambda X: X[:,0] / X[:,1] - 12 + c_noises[8]
		c9 = lambda X: (1.5 * X[:,5] + 1.9) / X[:,3] - 1 + c_noises[9]
		c10 = lambda X: (1.1 * X[:,6] + 1.9) / X[:,4] - 1 + c_noises[10]
		
		return  [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]
	@property
	def func_name(self):
		return "SpeedReducer"


class GasTransmission(Objective):
	def __init__(self, dim=4) -> None:
		super().__init__(dim)	
		self.min = np.array([20., 1., 20., 0.1])
		self.max = np.array([50., 10., 50., 60.])
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()	
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = np.random.normal(loc=0.0, scale=self.noise_stds[0])
		else:
			f_noise = 0
			
		x1, x2, x3, x4 = X[:,0], X[:,1], X[:,2], X[:,3]
		f = (
			8.61 * 10* np.sqrt(x1) * x2 * x3**(-2./3.) * x4**(-0.5) 
			+ 3.69 * x3 
			+ 7.72 * 1e4 * x1**(-1) * x2**(0.219) 
			- 765.43 * 1e2 * x1**(-1)  
		)
		f = (f - 173.9976) / (3553.9161 - 173.9976)
		f = (f - 0.5) * 2.0
		return  -f + f_noise
	
	def constraints(self, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [np.random.normal(loc=0.0, scale=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0]
		
		max_val = 99.3931
		min_val = 0.4373
			
		c0 = lambda X: 2*((X[:,3]/ (X[:,1]**(2)) + X[:,1]**(2) - 1) - min_val) / (max_val - min_val) - 0.5  + c_noises[0]
		return  [c0]
	@property
	def func_name(self):
		return "GasTransmission"


class WeldedBeamSO(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = np.array([0.125, 0.1, 0.1, 0.1])
		self.max  = np.array([10, 10, 10, 10])
		if self.noise_stds == None:
			self.noise_stds = self.noise_std_estimate()	
			print("Noise std:", self.noise_stds)
	def value(self, X, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			f_noise = np.random.normal(loc=0.0, scale=self.noise_stds[0])
		else:
			f_noise = 0

		target = 1.10471*X[:,0]**2*X[:,1] + 0.04811*X[:,2]*X[:,3]*(14+X[:,1])+ f_noise
		return target
	def constraints(self, is_noise=True):
		if is_noise:
			# First values in noise std array is for the main objective function 
			c_noises = [np.random.normal(loc=0.0, scale=c_std) for c_std in self.noise_stds[1:]]
		else:
			c_noises = [0,0,0,0,0,0]
		
		P = 6000.0
		L = 14.0
		E = 30e6
		G = 12e6
		t_max = 13600.0
		s_max = 30000.0
		d_max = 0.25

		def c0(X):
			t1 = P / (np.sqrt(2) * X[:,0] * X[:,1])
			M = P * (L + X[:,1] / 2)
			R = np.sqrt(0.25 * (X[:,1]**2 + (X[:,0] + X[:,2])**2))
			J = 2 * np.sqrt(2) * X[:,0] * X[:,1] * (X[:,1]**2 / 12 + 0.25 * (X[:,0] + X[:,2])**2)
			t2 = M * R / J
			t = np.sqrt(t1**2 + t1 * t2 * X[:,1] / R + t2**2)
			return t - t_max + c_noises[0]
		def c1(X):
			s = 6 * P * L / (X[:,3] * X[:,2]**2)
			return s - s_max + c_noises[1]
		
		c2 = lambda X: X[:,0] - X[:,3] + c_noises[2]
		
		c3 = lambda X: 0.10471 * X[:,0]**2 + 0.04811 * X[:,2] * X[:,3] * (14.0 + X[:,1]) - 5.0 + c_noises[3]
		
		def c4(X):
			d = 4 * P * L**3 / (E * X[:,2]**3 * X[:,3])
			return d - d_max + c_noises[4]
		def c5(X):
			P_c = (
			4.013
			* E
			* X[:,2]
			* X[:,3]**3
			* 6
			/ (L**2)
			* (1 - 0.25 * X[:,3] * np.sqrt(E / G) / L)
			)
			return P - P_c + c_noises[5]
		
		return  [c0,c1,c2,c3,c4,c5]
	@property
	def func_name(self):
		return "WeldedBeamSO"



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
		self.data = torch.FloatTensor([cv2.resize(x.numpy(), (5,5)) for x in self.data])
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
	def __init__(self, dim=25, trained_model_path='models_and_data/pretrained_models/trained_MNIST_acc_0.93.pth', gpu_id=0) -> None:
		self.min= torch.full([dim],-0.42421)
		self.max= torch.full([dim], 2.8215)
		self.noise_stds = [0,0]
		self.dim=dim
		self.device = torch.device(gpu_id if torch.cuda.is_available() else 'cpu')
		self.dataset = MNIST_Sensitive_Loader(root='models_and_data/mnist_data/', device=self.device, train=True, download=True)
		self.v0 , _ =  self.dataset.__getitem__(1504)
		if os.path.isfile(trained_model_path):
			self.model = MLP_MNIST(dropout=0.0, num_classes=10, n_hidden=10).to(self.device)
			self.model.load_state_dict(torch.load(trained_model_path))
			self.model.eval()
		else:
			print("Trained model is not existed")
		
	def value(self,X,is_noise=False):
		def grad_norm(X):
			X = (0.3081*X+0.1307)*255
			X = torch.FloatTensor(cv2.resize(X, (28,28))).to(self.device)
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
		return -grad_norm(X).cpu().numpy()
	def generate_features(self, sample_num, seed=1504):
		self.features = []
		max_pertubed = torch.full([self.dim], 0.5).to(self.device)
		min_pertubed = torch.full([self.dim], -0.4).to(self.device)
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
		return np.array(self.features.cpu(), dtype=np.float64)
	def constraints(self, is_noise=True):
		sz = int(np.sqrt(self.dim))
		ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
		c0 = lambda X: 0.8 - (ssim(torch.Tensor(X).view(X.shape[0], 1, sz, sz), self.v0.unsqueeze(0).cpu().repeat(X.shape[0],1,1,1))).numpy()
		return [c0]
		
	@property
	def func_name(self):
		return "SensitiveSample"





if __name__ == "__main__":
	problem = GasTransmission(dim=4)
	X = np.array([[20.2302,  5.0754, 20.8700,  0.1013]])
	C = problem.constraints(is_noise=False) 
	print(problem.value(X, is_noise=False))
	for c in C:
		print(c(X))