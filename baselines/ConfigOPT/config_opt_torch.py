import botorch
from botorch.acquisition import UpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize
import torch
import time

class ConfigOpt_torch():
	def __init__(self, cfg):
		self.dim = cfg.dimension
		self.n_iters = cfg.n_iters
		self.n_init = cfg.n_init
		self.n_raw_samples = cfg.n_raw_samples
		self.n_constraints = cfg.n_constraints
		self.X_train = None
		self.X_mean = None
		self.X_std = None
		self.target_Y_train = None
		self.constraints_Y_train = [None]*self.n_constraints
		self.beta = 2
		self.normalized_inputs = cfg.normalized_inputs
		self.use_cuda = cfg.use_cuda
		self.device = torch.device(0 if torch.cuda.is_available() and self.use_cuda else 'cpu')
		self.objective_type = cfg.objective_type
			
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
		
		# if self.normalized_inputs:
		# 	X_train = (self.X_train - X_mean)/X_std
		# else:
		# 	X_train = self.X_train
		
		for T in range(self.n_iters):
			if self.normalized_inputs:
				X_train = (self.X_train - X_mean)/X_std
			else:
				X_train = self.X_train
			
			gp_f = SingleTaskGP(X_train, standardize(self.target_Y_train).unsqueeze(1))
			mll = ExactMarginalLogLikelihood(gp_f.likelihood, gp_f)
			
			GP_constr = [SingleTaskGP(X_train, standardize(self.constraints_Y_train[i]).unsqueeze(1)) for i in range (self.n_constraints)]
			mll_constr = [ExactMarginalLogLikelihood(gp_ci.likelihood, gp_ci) for gp_ci in GP_constr]
			fit_gpytorch_model(mll)
			for i in range(self.n_constraints):
				fit_gpytorch_model(mll_constr[i])

			print(f"\n----------ConfigOpt - Optimization iteration {T+1}/{self.n_iters}----------\n")
			
			self.iteration = T+1
			X_cand = objective.generate_features(self.n_raw_samples, seed=int(time.time()%261504)).to(self.device)
			
			if self.normalized_inputs:
				X_cand = (X_cand - X_mean)/X_std
			
			LCB_g = []
			for i in range(self.n_constraints):
				# Calculate LCB
				LCB_g.append(-UpperConfidenceBound(GP_constr[i], self.beta, maximize=False)(X_cand.unsqueeze(1)))
			LCB_g = torch.stack(LCB_g)
			
			# Calculate LCB
			LCB_f = -UpperConfidenceBound(gp_f, self.beta, maximize=False)(X_cand.unsqueeze(1))
			
			check_feasibility = torch.prod(LCB_g <= 0, axis=0)
			feasible_idx = torch.where(check_feasibility == 1)[0]
			
			fea_LCB_f = LCB_f[feasible_idx.cpu().numpy()]
			
			if fea_LCB_f.shape[0] !=0:
				feasible_min_idx = torch.argmin(fea_LCB_f)
				X_next = X_cand[feasible_idx][feasible_min_idx]
			else:
				rand_idx = torch.randperm(self.n_raw_samples)[0]
				X_next = X_cand[rand_idx]

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

			print(f"\n**** Iteration [{T+1}/{self.n_iters}], current value = {true_value}, constraints {true_constraints}")
			

			
			optimal_values.append(true_value.item())
			constraint_values.append(true_constraints)
		return optimal_values, constraint_values


	