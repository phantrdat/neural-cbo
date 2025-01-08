from locale import normalize
from math import inf
import numpy as np
import pickle as pkl
from matplotlib  import pyplot as plt
import os
import torch
import itertools
from utils.objectives import *
import argparse
import json
import numpy as np
import time

from baselines.ConstrainedNeuralBO.neural_cbo import Neural_CBO
from baselines.ConfigOPT.config_opt_torch import ConfigOpt_torch
import importlib

from types import SimpleNamespace
import warnings

def fxn():
	warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	fxn()

DATA_DIR = 'data/'
RES_DIR = 'results'
default_json = 'objective_configs/SensitiveSample/configOpt_sensitive49.json'
parser = argparse.ArgumentParser()
parser.add_argument('-cfg', type=str, default=default_json, help='Config File')
parser.add_argument('-gpu_id', default=0, help='GPU ID')
parsed, unknown = parser.parse_known_args()

for arg in unknown:
    if arg.startswith(("-", "--")):
        # you can pass any arguments to add_argument
        parser.add_argument(arg.split('=')[0], type=str)
	
args = parser.parse_args()

GPU_ID = int(args.gpu_id)

def run_opt(alg, objective):
	t1 = time.time()
	
	optimal_values, constraint_values = alg.minimize(objective)
	
	t2 = time.time() - t1
	info = {'function_name': objective.func_name,
		 	"function_properties": objective.__dict__, 
			"optimal_values": optimal_values, 
			"constraint_values": constraint_values, 
			"X_train": alg.X_train.cpu(),
			"Y_train": alg.target_Y_train.cpu(),
			"Running time": t2}
	
	return info


if __name__ == '__main__':

	config_paths = args.cfg.split(',')
	print(f"Run {len(config_paths)} configs in total")
	for each_config in config_paths: 
		configs = json.load(open(each_config, "r"))
		print(configs)
		info = {"alg_configs": configs}
		configs = SimpleNamespace(**configs)
			
		
		objective = None
		objective_functions = importlib.import_module('utils.objectives')
		# if configs.objective_type =='synthetic':
		objective = getattr(objective_functions, configs.function_name)(dim=configs.dimension)

		
		
		# if 'constrainedneuralbo' == configs.algorithm_type.lower():		
		# 	print("Normalized outputs:", configs.normalized_outputs)
		# 	print("Normalized inputs:", configs.normalized_inputs)
		# 	for run_idx in range(configs.first_run, configs.last_run):
		# 		print("Run:", run_idx)
				
		# 		constrained_neuralbo = Constrained_NeuralBO(cfg=configs)
		# 		main_info = run_opt(constrained_neuralbo, objective)
		# 		info.update(main_info)
		# 		save_root = f"results/{info['function_name']}_DIM_{configs.dimension}_ITERS_{configs.n_iters}/{configs.algorithm_type}"
		# 		if os.path.isdir(save_root) == False:
		# 			os.makedirs(save_root)

		# 		file_name = f"{save_root}/{configs.algorithm_type}_{info['function_name']}_dim{info['function_properties']['dim']}.{run_idx}.pkl"
		# 		pkl.dump(info, open(file_name,'wb'))
				
		# if 'constrained_neuralbo_ei_conditioned' == configs.algorithm_type.lower():		
		# 	print("Normalized outputs:", configs.normalized_outputs)
		# 	print("Normalized inputs:", configs.normalized_inputs)
		# 	for run_idx in range(configs.first_run, configs.last_run):
		# 		print("Run:", run_idx)
				
		# 		constrained_neuralbo = Constrained_NeuralBO_EI_Conditioned(cfg=configs)
		# 		main_info = run_opt(constrained_neuralbo, objective)
		# 		info.update(main_info)
		# 		save_root = f"results/{info['function_name']}_DIM_{configs.dimension}_ITERS_{configs.n_iters}/{configs.algorithm_type}"
		# 		if os.path.isdir(save_root) ==False:
		# 			os.makedirs(save_root)
		# 			print(save_root)

		# 		file_name = f"{save_root}/{configs.algorithm_type}_{info['function_name']}_dim{info['function_properties']['dim']}.{run_idx}.pkl"
		# 		pkl.dump(info, open(file_name,'wb'))  
		

		# if 'constrained_neuralbo_ei' == configs.algorithm_type.lower():		
		# 	print("Normalized outputs:", configs.normalized_outputs)
		# 	print("Normalized inputs:", configs.normalized_inputs)
		# 	for run_idx in range(configs.first_run, configs.last_run):
		# 		print("Run:", run_idx)
				
		# 		constrained_neuralbo_ei = Constrained_Neural_EI(cfg=configs)
		# 		main_info = run_opt(constrained_neuralbo_ei, objective)
		# 		info.update(main_info)
		# 		save_root = f"results/{info['function_name']}_DIM_{configs.dimension}_ITERS_{configs.n_iters}/{configs.algorithm_type}"
		# 		if os.path.isdir(save_root) ==False:
		# 			os.makedirs(save_root)

		# 		file_name = f"{save_root}/{configs.algorithm_type}_{info['function_name']}_dim{info['function_properties']['dim']}.{run_idx}.pkl"
		# 		pkl.dump(info, open(file_name,'wb'))
		
		# if 'noiseless_neuralei_with_lcb_constraints' == configs.algorithm_type.lower():		
		# 	print("Normalized outputs:", configs.normalized_outputs)
		# 	print("Normalized inputs:", configs.normalized_inputs)
		# 	for run_idx in range(configs.first_run, configs.last_run):
		# 		print("Run:", run_idx)
				
		# 		constrained_neuralbo = Noiseless_NeuralEI_With_LCB_Constraints(cfg=configs)
		# 		main_info = run_opt(constrained_neuralbo, objective)
		# 		info.update(main_info)
		# 		save_root = f"results/{info['function_name']}_DIM_{configs.dimension}_ITERS_{configs.n_iters}/{configs.algorithm_type}"
		# 		if os.path.isdir(save_root) ==False:
		# 			os.makedirs(save_root)
		# 			print(save_root)

		# 		file_name = f"{save_root}/{configs.algorithm_type}_{info['function_name']}_dim{info['function_properties']['dim']}.{run_idx}.pkl"
		# 		pkl.dump(info, open(file_name,'wb'))  
		
		if 'neural-cbo' == configs.algorithm_type.lower():		
			print("Normalized outputs:", configs.normalized_outputs)
			print("Normalized inputs:", configs.normalized_inputs)
			for run_idx in range(configs.first_run, configs.last_run):
				print("Run:", run_idx)
				
				noisy_neuralei = Neural_CBO(cfg=configs)
				main_info = run_opt(noisy_neuralei, objective)
				info.update(main_info)
				save_root = f"results/{info['function_name']}_DIM_{configs.dimension}_ITERS_{configs.n_iters}/{configs.algorithm_type}"
				if os.path.isdir(save_root) ==False:
					os.makedirs(save_root)
					print(save_root)

				file_name = f"{save_root}/{configs.algorithm_type}_{info['function_name']}_dim{info['function_properties']['dim']}.{run_idx}.pkl"
				pkl.dump(info, open(file_name,'wb'))  
		
		if 'configopt' == configs.algorithm_type.lower():	
			print("Normalized outputs:", configs.normalized_outputs)
			print("Normalized inputs:", configs.normalized_inputs)
			for run_idx in range(configs.first_run, configs.last_run):
				print("Run:", run_idx)
				
				config_opt_torch = ConfigOpt_torch(cfg=configs)
				main_info = run_opt(config_opt_torch, objective)
				info.update(main_info)
				save_root = f"results/{info['function_name']}_DIM_{configs.dimension}_ITERS_{configs.n_iters}/{configs.algorithm_type}"
				if os.path.isdir(save_root) ==False:
					os.makedirs(save_root)
					print(save_root)

				file_name = f"{save_root}/{configs.algorithm_type}_{info['function_name']}_dim{info['function_properties']['dim']}.{run_idx}.pkl"
				pkl.dump(info, open(file_name,'wb'))  
		print(f"Finished {each_config} !!!!!!!")
		

	
