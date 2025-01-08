import os


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
import torch.nn as nn
from utils.base_networks import BaseNN
from torch.distributions import Normal
import math
import time
import copy
from torch.func import functional_call, vmap, grad
from botorch.utils.probability.utils import (
	ndtr as Phi,
	phi,
)
class Neural_CBO():
	"""Constrained BO using Thompson Sampling with Deep Neural Networks. """
	def __init__(self, cfg):
		
		self.dim = cfg.dimension
		self.n_iters = cfg.n_iters
		self.n_constraints = cfg.n_constraints
		self.activation = cfg.activation
		
		# ****** Hyper-parameters for target objective model ******
		# L2 regularization strength
		self.target_weight_decay = cfg.target_weight_decay
		# hidden size of the NN layers
		self.target_hidden_size = cfg.target_hidden_size
		# number of layers
		self.target_n_layers = cfg.target_n_layers
		# NN hyper-parameters
		self.target_learning_rate = cfg.target_learning_rate
		# Training epochs for objective model
		self.target_epochs = cfg.target_epochs


		# ****** Hyper-parameters for constraint models ******
		# L2 regularization strength
		self.constraints_weight_decay = cfg.constraints_weight_decay
		# hidden size of the NN layers
		self.constraints_hidden_sizes = cfg.constraints_hidden_sizes
		# number of layers
		self.constraints_n_layers = cfg.constraints_n_layers
		# NN hyper-parameters
		self.constraints_learning_rates = cfg.constraints_learning_rates
		# Training epochs for constraint models
		self.constraints_epochs = cfg.constraints_epochs


		self.update_cycle = cfg.update_cycle

		self.use_cuda = cfg.use_cuda
		self.device = torch.device(0 if torch.cuda.is_available() and self.use_cuda else 'cpu')
		

		# Init neural network model for  target objective 
		self.target_model = BaseNN(input_size=self.dim,
						   hidden_size=self.target_hidden_size,
						   n_layers=self.target_n_layers,
						   p=0.0, activation=self.activation).to(self.device)

		
  
		# Init neural network models for constraints
		self.constraint_models  = [BaseNN(input_size=self.dim, hidden_size=self.constraints_hidden_sizes[i], 
									n_layers=self.constraints_n_layers[i], p=0.0, activation=self.activation).to(self.device) 
								for i in range (self.n_constraints)]
		
		self.target_optimizer = torch.optim.Adam(self.target_model.parameters(), lr=self.target_learning_rate, weight_decay=self.target_weight_decay)
		
		self.constraint_optimizers = [
			torch.optim.Adam(self.constraint_models[i].parameters(), lr=self.constraints_learning_rates[i], weight_decay=self.constraints_weight_decay[i])
					for i in range (self.n_constraints)
		]


		# if cfg.use_lr_scheduler:
		# 	self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
		# else:
		# 	self.scheduler = None
			
		self.iteration = 0
			
		self.normalized_inputs = cfg.normalized_inputs
		self.normalized_outputs = cfg.normalized_outputs
		
		
		# Algorithm objective optimization setting
		self.objective_type = cfg.objective_type
		self.n_init = cfg.n_init
		self.n_raw_samples = cfg.n_raw_samples

		# Algorithm specific configs


		self.target_U_inv = (torch.eye(self.approximator_target_dim)/self.target_weight_decay).double().to(self.device)
		self.constraints_U_inv = [(torch.eye(self.approximator_constraints_dim[i])/self.constraints_weight_decay[i]).double().to(self.device) 
							for i in range(self.n_constraints)]
		# self.R = 1
		self.adaptive_exploration = cfg.adaptive_exploration
		self.target_exploration_coeff = cfg.target_exploration_coeff
		self.constraints_exploration_coeff = cfg.constraints_exploration_coeff
		self.f_best = None
		self.X_train = None
		self.X_mean = None
		self.X_std = None
		self.target_Y_train = None
		self.constraints_Y_train = [None]*self.n_constraints
	@property
	def approximator_target_dim(self):
		"""Sum of the dimensions of all trainable layers in the network.
		"""
		return sum(w.numel() for w in self.target_model.parameters() if w.requires_grad)
	
	@property
	def approximator_constraints_dim(self):
		"""Sum of the dimensions of all trainable layers in the network.
		"""
		return [sum(w.numel() for w in cm.parameters() if w.requires_grad) for cm in self.constraint_models]
	

	
	def random_seed(self):
		frac, whole = math.modf(time.time())
		try:
			seed = int(whole/(10000*frac)) + self.iteration
			torch.manual_seed(seed)
		except:
			torch.manual_seed(1111)

	
	def approx_grad(self, model, x):
			params = {k: v.detach() for k, v in model.named_parameters()}
			buffers = {k: v.detach() for k, v in model.named_buffers()}
			def compute_preds(params, buffers, sample):
				preds = functional_call(model, (params, buffers), (sample,))
				return torch.sum(preds, dim=0)
			ft_compute_grad = grad(compute_preds)
			ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0))
			ft_per_sample_grads = ft_compute_sample_grad(params, buffers, x)
		
			grads_approx_fast = []
			all_grads = list(ft_per_sample_grads.values())
			for i in range(len(all_grads)):
				all_grads[i] = all_grads[i].flatten(1)
			grads_approx_fast = torch.cat(all_grads, dim=1)

			return grads_approx_fast

	def calculate_gradients_fast(self, x):
		# target_grads_approx = self.approx_grad(self.target_model, x)/np.sqrt(self.target_hidden_size)
		# constraint_grads_approx = [self.approx_grad(self.constraint_models[i], x)/np.sqrt(self.constraints_hidden_sizes[i]) for i in range(self.n_constraints)]
		
		target_grads_approx = self.approx_grad(self.target_model, x)/math.sqrt(self.target_hidden_size)
		constraint_grads_approx = [self.approx_grad(self.constraint_models[i], x)/math.sqrt(self.constraints_hidden_sizes[i]) for i in range(self.n_constraints)]

		return target_grads_approx, constraint_grads_approx
	

	

	def predict(self, X):
		"""Predict reward.
		"""
		# eval mode
		self.target_model.eval()
		for cm in self.constraint_models:
			cm.eval()
		
		constraint_preds = [cm(X).detach().squeeze() for cm in self.constraint_models] 
		return self.target_model.forward(X).detach().squeeze(), constraint_preds
	
	def posterior_std(self, X):
		target_grads_approx, constraint_grads_approx = self.calculate_gradients_fast(X)
		def posterior_cov(features, U_inv):
			covariances = features @ U_inv @ features.T
			covariances = torch.diag(covariances)			
			return covariances
	
		target_var = self.target_weight_decay*posterior_cov(target_grads_approx, self.target_U_inv)
		target_std = self.target_exploration_coeff*torch.sqrt(target_var)

		constraints_var = [self.constraints_weight_decay[i] *posterior_cov(constraint_grads_approx[i], self.constraints_U_inv[i]) 
					for i in range(self.n_constraints)]
		constraints_std = [beta_c*torch.sqrt(c_var) for (c_var, beta_c) in zip(constraints_var, self.constraints_exploration_coeff)]
		return target_std, constraints_std
		
	def constrained_neural_ei(self, X):
		target_posterior_mu, constraints_posterior_mu = self.predict(X)
		target_posterior_std, constraints_posterior_std = self.posterior_std(X)
		
		# EI for target
		dist = Normal(0, 1)
		z = (self.f_best - target_posterior_mu)/target_posterior_std
		
		# wEI = sigma*(alpha*z*Phi(z) + (1-alpha))
		EI_f  = target_posterior_std*(z*Phi(z) + phi(z))
		LCB_g = []
		beta = 0.5
		for i in range(self.n_constraints):
			LCB_g.append(constraints_posterior_mu[i] - beta*constraints_posterior_std[i])
			
		LCB_g = torch.stack(LCB_g)
		# PoF = torch.ones(X.shape[0]).to(self.device)
		# for (c_mu, c_std) in zip (constraints_posterior_mu, constraints_posterior_std):
			# PoF *= dist.cdf(-c_mu/c_std)
		# 
		
		return EI_f, LCB_g, target_posterior_mu, constraints_posterior_mu, target_posterior_std, constraints_posterior_std

	

	
	def train(self, x_train, y_train, constrained_y_train):
		"""Train neural approximator."""
		# train mode
		x_train = x_train.double()
		y_train = y_train.double()
		self.target_model.train()
		for cm in self.constraint_models:
			cm.train()
		def train_model(model, x_train, y_train, optimizer, n_epochs):
			model.train()
			loss = torch.tensor(0.0)
			for i in (range(n_epochs)):
				y_pred = model.forward(x_train).squeeze().double() 
				loss = nn.MSELoss()(y_pred, y_train).double()
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			return loss
		
		target_loss = train_model(self.target_model, x_train, y_train, self.target_optimizer, self.target_epochs)
		
		constraint_losses = [train_model(cm, x_train, c_y_train, opt, n_epochs) for (cm, opt, c_y_train, n_epochs) 
					   in zip(self.constraint_models, self.constraint_optimizers, constrained_y_train, self.constraints_epochs)]
		
		return target_loss, constraint_losses	

	def test_mse(self, x_test, target_y_test, constraints_y_test):
		self.target_model.eval()
		for c in self.constraint_models:
			c.eval()
		
		
		target_Y_pred = self.target_model(x_test)
		constraint_Y_preds = [c(x_test) for c in self.constraint_models]
		print(f"- Eval target model MSE at iteration {self.iteration}:", nn.MSELoss()(target_y_test, target_Y_pred.squeeze()).double().item())
		print(f"\n- Eval constraints models MSE at iteration {self.iteration}:", [nn.MSELoss()(y_true, y_pred.squeeze()).double().item() 
																	   for (y_true, y_pred) in zip(constraints_y_test, constraint_Y_preds)])
	
	def update_A_inv(self, x_t):

		# Implement matrix inversion with Sherman-Morrison method # 
	
		def inv_sherman_morrison(u, A_inv):
			"""Inverse of a matrix with rank 1 update.
			"""
			Au = torch.matmul(A_inv, u.T).squeeze(-1)
			A_inv -= torch.outer(Au, Au)/(1+torch.matmul(u, Au))
			return A_inv
	
		target_xt_grads_approx, constraint_xt_grads_approx = self.calculate_gradients_fast(x_t.unsqueeze(0))
		self.target_U_inv = inv_sherman_morrison(target_xt_grads_approx, self.target_U_inv)
		for i in range(self.n_constraints):
			self.constraints_U_inv[i] =  inv_sherman_morrison(constraint_xt_grads_approx[i], self.constraints_U_inv[i])
	
			
	

	def minimize(self, objective):
		init_points = objective.generate_samples(self.n_init, is_noise=True) #Using fixed seed in utils/objective.py
		print("Initial points:", init_points)
		self.X_train = init_points['features'].to(self.device)
		self.target_Y_train = init_points['observations'].to(self.device)
		self.constraints_Y_train =  init_points['constraint_observations'].to(self.device)
		constraint_values = [] 
		# Pick the last point of init_points to draw plot
		
		
		optimal_values = [objective.value(self.X_train[-1], is_noise=False).item()]
		constraint_values.append([c.item() for c in  objective.constraints(self.X_train[-1], is_noise=False)])
		
		check_feasible_init = torch.prod(self.constraints_Y_train <= 0, axis=0)
		if torch.any(check_feasible_init):
			init_feasible_idx = torch.where(check_feasible_init == 1)[0]
			self.f_best = torch.min(self.target_Y_train[init_feasible_idx])
		else:
			self.f_best = torch.tensor(1e12)
			

		objective.max = objective.max.to(self.device)
		objective.min = objective.min.to(self.device)
		X_mean = (objective.max + objective.min)/2
		X_std = torch.abs(objective.max  - X_mean)

		self.X_mean = X_mean
		self.X_std = X_std

		# Generate data to test models at each optimization iterations
		x_test = objective.generate_features(10000, seed=0).to(self.device)
		target_y_test = torch.Tensor([objective.value(x, is_noise=False) for x in x_test]).to(self.device)

		constraints_y_test = torch.vstack([torch.FloatTensor(objective.constraints(x, is_noise=False)).unsqueeze(0) for x in x_test])
		constraints_y_test = constraints_y_test.T.to(self.device)
		if self.normalized_inputs:
			x_test = (x_test- X_mean)/X_std
		
		if len(self.X_train) != 0 and len(self.target_Y_train)!=0:

			# print("**Fitting known dataset**")
			if self.normalized_inputs:
				X_train = (self.X_train - X_mean)/X_std
			else:
				X_train = self.X_train


			self.train(X_train, self.target_Y_train, self.constraints_Y_train)
		
		# Test initial models:
		self.test_mse(x_test, target_y_test, constraints_y_test)
		


		for T in range(self.n_iters):
			print(f"\n----------Constrained NeuralBO - Optimization iteration {T+1}/{self.n_iters}----------\n")
			self.iteration = T+1

			# if self.adaptive_exploration:
				
				# target_exploration_coeff, constraints_exploration_coeff = self.estimate_maximum_IG(objective)
				# self.target_exploration_coeff = self.R*torch.sqrt(2*target_exploration_coeff +2) 
				# self.constraints_exploration_coeff = [self.R*torch.sqrt(2*beta_c +2) for beta_c in constraints_exploration_coeff]
			
				# print(f"Beta target {self.target_exploration_coeff}, Beta constraints: {self.constraints_exploration_coeff}\n")
			
			
			self.random_seed()

			self.target_model.eval()
			for cm in self.constraint_models:
				cm.eval()
			
			target_mu_xt = 0
			constraints_mu_xt = None
			target_std_xt = 0
			constraints_std_xt = None
			# acqf_target_xt = 0 
			# acqf_constraints_xt = None
			# is_feasible = False
			X_next = None

			### Approach 1

			# while is_feasible == False:
			X_cand = objective.generate_features(self.n_raw_samples, seed=int(time.time()%261504)).to(self.device)
			if self.normalized_inputs:
				X_cand = (X_cand - X_mean)/X_std
			EI_f, LCB_g, f_mu, cs_mu, f_std, cs_std = self.constrained_neural_ei(X_cand)

			check_feasibility = torch.prod(LCB_g <= 0, axis=0)
			feasible_idx = torch.where(check_feasibility == 1)[0]
			
			fea_EI_f = EI_f[feasible_idx.cpu().numpy()]
			# fea_EI_g = LCB_g[:, feasible_idx.cpu().numpy()]
				
			
			# for c_i in c_hat:
			# 	acqf_values+=c_i
			if fea_EI_f.shape[0] !=0:
				feasible_min_idx = torch.argmax(fea_EI_f)
				target_mu_xt, constraints_mu_xt = f_mu[feasible_idx][feasible_min_idx], [c[feasible_idx][feasible_min_idx].item() for c in cs_mu]
				target_std_xt, constraints_std_xt =  f_std[feasible_idx][feasible_min_idx], [c[feasible_idx][feasible_min_idx].item() for c in cs_std]
				X_next = X_cand[feasible_idx][feasible_min_idx]

			else:
				rand_idx = torch.randperm(self.n_raw_samples)[0]
				X_next = X_cand[rand_idx]
			
			self.update_A_inv(X_next)

			if self.normalized_inputs:
				X_next = (X_next*X_std + X_mean).detach()
			else:
				X_next = X_next.detach()

			
			self.X_train = torch.cat([self.X_train, X_next.clone().unsqueeze(0)])
			
			# Noisy observations
			target_observation = objective.value(X_next, is_noise=True).to(self.device)
			constraint_observations =  [c.to(self.device) for c in objective.constraints(X_next, is_noise=True)]
			
			if self.objective_type == 'synthetic':
				true_value = objective.value(X_next, is_noise=False)
				true_constraints = [c.item() for c in  objective.constraints(X_next, is_noise=False)]
			else:
				true_value = target_observation
				true_constraints = constraint_observations
			self.target_Y_train = torch.cat([self.target_Y_train, target_observation.unsqueeze(0)])
			
			self.constraints_Y_train = torch.hstack([self.constraints_Y_train, torch.FloatTensor(constraint_observations).unsqueeze(0).T.to(self.device)])
			if torch.all(torch.FloatTensor(constraint_observations) <=0):
				self.f_best = torch.min(self.f_best, target_mu_xt)
			
			print("function best:", self.f_best)
			if (T+1) % self.update_cycle == 0:
				# print(f"** Train Constrained NeuralBO with {self.epochs} epochs")
				if self.normalized_inputs:				
					X_train = (self.X_train - X_mean)/X_std
				else:
					X_train = self.X_train
				
				target_loss, constraint_losses = self.train(X_train, self.target_Y_train, self.constraints_Y_train)
			
			# Test mse at ith iteration for models:
			self.test_mse(x_test, target_y_test, constraints_y_test)
			
			print(f"\n**** Iteration [{T+1}/{self.n_iters}], current value = {true_value}, constraints {true_constraints}")
			print(f"\n**** Iteration [{T+1}/{self.n_iters}], mu_target = {target_mu_xt}, mu_constraints {constraints_mu_xt}")
			
			print(f"\n**** Iteration [{T+1}/{self.n_iters}], std_target = {target_std_xt}, std_constraints {constraints_std_xt}")

			
			optimal_values.append(true_value.item())
			constraint_values.append(true_constraints)

		return optimal_values, constraint_values
	