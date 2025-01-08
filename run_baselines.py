# ***** This code provide an wrapper to run baselines provided in CONFIGOpt paper 
# ***** Link: https://proceedings.mlr.press/v202/xu23h/xu23h.pdf 



import os
import numpy as np
from baselines.ConfigOPT import config
import matplotlib.pyplot as plt
from types import SimpleNamespace
import json
import importlib
import GPy
import time
import pickle as pkl
import baselines.ConfigOPT.safeopt as safeopt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse

# df = 'objective_configs/Gramacy/ConfigOpt_gramacy2.json'
# df = 'objective_configs/Gardner/ConfigOpt_gardner2.json'
# df = 'objective_configs/Gardner/cEI_gardner2.json'
df = 'objective_configs/SensitiveSample/cEI_sensitive49.json'
# df = 'objective_configs/Gomez_and_Levy/ConfigOpt_gomez_and_levy2.json'
parser = argparse.ArgumentParser()
parser.add_argument('-cfg', type=str, default=df, help='Config File')
args = parser.parse_args()


MODULES_IMPORT_PATH = 'utils.objectives_numpy'
OBJECTIVE_FUNCTIONS = importlib.import_module(MODULES_IMPORT_PATH)

class BaselineWrapper:
	def __init__(self, obj_cfgs_dict) -> None:
		self.obj_cfgs_dict = obj_cfgs_dict
		self.obj_cfgs = SimpleNamespace(**self.obj_cfgs_dict)
		
		self.optimizer_type = self.obj_cfgs.algorithm_type
		
		self.objective = getattr(OBJECTIVE_FUNCTIONS, self.obj_cfgs.function_name)(dim=self.obj_cfgs.dimension)
		self.base_opt_config = {
			'noise_level':self.objective.noise_stds,
			'total_eval_num': self.obj_cfgs.n_iters,
			}
		self.problem_config = None
		self.opt = None 
		self.best_obj_list = []
		self.total_cost_list = []
	def get_config(self):
		
		self.problem_config = dict()
		self.problem_config['problem_name'] = self.obj_cfgs.function_name
		# self.problem_config['f_min'] = self.obj_cfgs.f_min
		self.problem_config['var_dim'] = self.obj_cfgs.dimension
		self.problem_config['discretize_num_list'] = [self.obj_cfgs.discretize_num for _ in range(self.problem_config['var_dim'])]
		self.problem_config['num_constrs'] = 1
		self.problem_config['bounds'] = np.stack([self.objective.min, self.objective.max]).T.tolist()
		self.problem_config['train_X'] = self.objective.generate_features(self.obj_cfgs.n_raw_samples, seed=int(time.time()/261504)) 
		# problem_config['train_X'] = safeopt.linearly_spaced_combinations(
		# 		problem_config['bounds'],
		# 		[20 for _ in range(problem_config['var_dim'])]
		# 	)
		
		self.problem_config['parameter_set'] = self.problem_config['train_X']
		self.problem_config['eval_simu'] = False
		self.problem_config['obj'] = self.objective.value
		self.problem_config['constrs_list'] = self.objective.constraints()
		self.problem_config['init_safe_points'] = self.objective.generate_features(500) #Using fixed seed in utils/objective.py
		self.problem_config['init_points'] =  self.objective.generate_features(self.obj_cfgs.n_init) 
		self.problem_config['y_init'] = self.problem_config['obj'](self.problem_config['init_points'], is_noise=False)
		

		if self.obj_cfgs.gp_kernel.lower() == 'gaussian':
			kernel = GPy.kern.RBF(input_dim=self.problem_config['var_dim'], variance=2.,
								  lengthscale=1.0, ARD=True)
		if self.obj_cfgs.gp_kernel.lower()  == 'poly':
			kernel = GPy.kern.Poly(input_dim=self.problem_config['var_dim'],variance=2.0, 
						  		   scale=1.0, order=1)
		if self.obj_cfgs.gp_kernel.lower()  == 'matern':
		# kernel = GPy.kern.RBF(input_dim=, variance=1,
		# 						  lengthscale=1, ARD=True)
			kernel = GPy.kern.Matern52(input_dim=self.problem_config['var_dim'], variance=1, lengthscale=1)
		self.problem_config['kernel'] = [kernel, kernel.copy()]
		


	def get_optimizer(self):
		optimizer_config = self.base_opt_config.copy()
		problem = config.OptimizationProblem(self.problem_config)
		
		 
		if self.optimizer_type.lower() == 'safe_bo':
			self.opt = config.SafeBO(problem, optimizer_config)
			self.best_obj_list = [-self.optopt.best_obj]
		if self.optimizer_type.lower() == 'cei':
			self.opt = config.ConstrainedEI(problem, optimizer_config)
			self.best_obj_list = [self.opt.best_obj]
		if self.optimizer_type.lower() == 'configopt':
			self.opt = config.CONFIGOpt(problem, optimizer_config)
			self.best_obj_list = [self.opt.best_obj]
		
	
	def run(self):
		t1 = time.time()
		self.get_config()
		self.get_optimizer()
		x_plot = np.expand_dims(self.problem_config['init_points'][-1], axis=0)
		optimal_values = [self.objective.value(x_plot, is_noise=False).item()]
		constraint_values = [[c(x_plot).item() for c in self.objective.constraints(is_noise=False)]]
		
		for _ in range(self.obj_cfgs.n_iters):
			self.opt.parameter_set = self.objective.generate_features(self.obj_cfgs.n_raw_samples, seed=int(time.time())%261504)
			x_next, y_obj, constr_vals = self.opt.make_step()
			# print(y_obj, constr_vals)
			optimal_values.append(self.objective.value(x_next, is_noise=False).item())
			constraint_values.append([c(x_next).item() for c in self.objective.constraints(is_noise=False)])
		t2 = time.time() - t1
		info = {'function_name': self.objective.func_name,
		 	"function_properties": self.objective.__dict__, 
			"optimal_values": optimal_values, 
			"alg_configs": self.obj_cfgs_dict,
			"constraint_values": constraint_values, 
			"Running time": t2}
		return info	
	 

	
	
	
	

# def get_con_regret(obj_list, constr_list):
# 	obj_arr = np.array(obj_list)
# 	constr_arr = np.array(constr_list)
	

# 	# pos_regret_arr = np.maximum(obj_arr-problem_config['f_min'],0)
# 	is_infeasible =  [(np.array(v)>0).any() for v in constr_arr]
# 	obj_arr[is_infeasible] = np.inf
# 	print("Feasible min:", np.min(obj_arr))
# 	min_feasible_value_found_so_far = [np.min(obj_arr[:i]) for i in range(1, len(obj_arr))]
# 	pos_constr_arr = np.maximum(constr_arr, 0)
# 	all_constr_arr = np.sum(pos_constr_arr, axis=1)
	
# 	return min_feasible_value_found_so_far, np.minimum.accumulate(all_constr_arr)


if __name__ == '__main__':
	
	cfg_path = args.cfg 
	obj_cfgs_dict = json.load(open(cfg_path, "r"))
	run_idx_range = range(obj_cfgs_dict['first_run'], obj_cfgs_dict['last_run'])
	
	for run_idx in run_idx_range:
		alg = BaselineWrapper(obj_cfgs_dict)
		info = alg.run()
		config_opt_obj_list = info['optimal_values']
		config_opt_constr_list = info['constraint_values']
		
		# config_min_value_found_so_far, config_constr_regret = get_con_regret(config_opt_obj_list, config_opt_constr_list)
		
		save_root = f"results/{info['function_name']}_DIM_{info['alg_configs']['dimension']}_ITERS_{info['alg_configs']['n_iters']}/{info['alg_configs']['algorithm_type']}"
		if os.path.isdir(save_root) ==False:
			os.makedirs(save_root)
		file_name = f"{save_root}/{info['alg_configs']['algorithm_type']}_{info['function_name']}_dim{info['alg_configs']['dimension']}.{run_idx}.pkl"
		print(file_name)
		pkl.dump(info, open(file_name,'wb'))
	
	# config_opt_obj_list = info['optimal_values']
	# config_opt_constr_list = info['constraint_values']
	
	# config_min_value_found_so_far, config_constr_regret = get_con_regret(config_opt_obj_list, config_opt_constr_list)

	# safe_min_value_found_so_far, safe_constr_regret = get_con_regret(safe_opt_obj_list, safe_opt_constr_list)

	# cei_min_value_found_so_far, cei_constr_regret = get_con_regret(cei_opt_obj_list, cei_opt_constr_list)
	
	# config_constr_regret = get_con_regret(config_opt_obj_list, config_opt_constr_list)

	# safe_constr_regret = get_con_regret(safe_opt_obj_list, safe_opt_constr_list)

	# cei_constr_regret = get_con_regret(cei_opt_obj_list, cei_opt_constr_list)
 
	# plt.plot(config_constr_regret, label='CONFIG')
	# plt.legend()
	# plt.savefig('regret.png')
	# plt.clf()
	

	# plt.plot(config_min_value_found_so_far, label='CONFIG')
	# plt.legend()
	# plt.savefig('values.png')