import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import pickle as pkl
import numpy as np
import os
import glob
from objectives import *
import re
import matplotlib.patches as patches
import copy
import random
from tqdm import tqdm
from tueplots import bundles, axes
def break_camel_case(text):
    # This regular expression finds positions in the string where a lowercase letter is followed by an uppercase letter
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

TARGET_INFO = { 
	"Simionescu": {"DIM": 2, "N_ITERS": 100, "N_TRIALS": 10 },
	"Ackley": {"DIM": 5, "N_ITERS": 200, "N_TRIALS": 10 },
	"Gomez_and_Levy": {"DIM": 2, "N_ITERS": 100, "N_TRIALS": 10 },
	"Hartmann":  {"DIM": 6, "N_ITERS": 200, "N_TRIALS": 10},
	"Branin": {"DIM": 2, "N_ITERS": 100, "N_TRIALS": 10},
	"Gardner": {"DIM": 2, "N_ITERS": 100, "N_TRIALS": 10 },
	"Gramacy": {"DIM": 2, "N_ITERS": 100, "N_TRIALS": 10 },
	"SpeedReducer": {"DIM": 7, "N_ITERS": 100, "N_TRIALS": 10},
	"GasTransmission": {"DIM": 4, "N_ITERS": 100, "N_TRIALS": 10}, 
	"SensitiveSample": {"DIM": 49, "N_ITERS": 200, "N_TRIALS": 10}, 
	}
FIXED_COLORS = ['blue', 'red', 'green', 'black', 'purple', 'olive', 'cyan', 'brown']
FIXED_MARKERS = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X']
STD_SCALE = 0.5
# Help func

def subplots_centered(nrows, ncols, figsize, nfigs):
	"""
	Modification of matplotlib plt.subplots(),
	useful when some subplots are empty.
	
	It returns a grid where the plots
	in the **last** row are centered.
	
	Inputs
	------
		nrows, ncols, figsize: same as plt.subplots()
		nfigs: real number of figures
	"""
	assert nfigs < nrows * ncols, "No empty subplots, use normal plt.subplots() instead"
	
	fig = plt.figure(figsize=figsize)
	axs = []
	
	m = nfigs % ncols
	m = range(1, ncols+1)[-m]  # subdivision of columns
	gs = gridspec.GridSpec(nrows, m*ncols)

	for i in range(0, nfigs):
		row = i // ncols
		col = i % ncols

		if row == nrows-1: # center only last row
			off = int(m * (ncols - nfigs % ncols) / 2)
		else:
			off = 0

		ax = plt.subplot(gs[row, m*col + off : m*(col+1) + off])
		axs.append(ax)
		
	return fig, axs

class PlotSingleRegret:
	def __init__(self, dim, n_iters, func_name, result_dir, list_colors, list_markers, list_algs, n_trials=10, minimize=True, log_plot=False) -> None:
		self._dim = dim
		self._n_iters = n_iters
		self._func_name = func_name
		self._minimize = minimize
		self._log_plot = log_plot
		self._result_dir = os.path.join(result_dir, f"{func_name}_DIM_{dim}_ITERS_{n_iters}/")
		self._n_trials = n_trials
		self._colors = list_colors
		self._markers = list_markers
		self._algs = list_algs
	def assign_color_and_marker(self):
		assert (len(self._algs) <= len(self._colors)) and (len(self._algs) <= len(self._markers)), \
		"The numbers of colors or markers must be larger than the number of algorithms"
		color_map = {}
		marker_map = {}
		for i, alg in enumerate(self._algs):
			color_map[alg] = self._colors[i]
			marker_map[alg] = self._markers[i]
		return color_map, marker_map
	def plot_single_regret(self, from_iter=None, to_iter=None):

		colors_map, markers_map = self.assign_color_and_marker()
		plotting_objs = {'regret_mean':[],
				   		 'regret_std':[],
						 'feasible_minimum_mean':[],
				   		 'feasible_minimum_std':[], 
						 'violation_mean':[], 
						 'violation_std':[],
						 'rv_mean':[],
						 'rv_std':[],
						 'min_mean':[],
						 'min_std':[],
						 'colors': [], 
						 'labels':[], 
						 'markers':[]}
		for alg in self._algs: 	
			print(alg)
			res_files = glob.glob(f'{self._result_dir}/{alg}/*')

			all_cummu_regrets = np.empty((0, self._n_iters+1))
			all_violations =  np.empty((0, self._n_iters+1))
			all_feasible_minimum = np.empty((0, self._n_iters))
			all_regret_violation_sum = np.empty((0, self._n_iters+1))
			all_min_values = np.empty((0, self._n_iters+1))
			for pkl_file in sorted(res_files[:self._n_trials]):
				
				Di  = pkl.load(open(pkl_file,'rb'))	
				best_feasible_y = dict(Di['alg_configs'])['best_feasible_y']
				all_optimal_values = np.array(Di['optimal_values'])
				all_constraints = np.array(Di['constraint_values'])
				
				
				simutaneous_regrets = np.maximum(0, all_optimal_values - best_feasible_y)
				# best_simutaneous_regrets_so_far = [np.min(simutaneous_regrets[:i]) for i in range(1, self._n_iters + 2)]
				cummulative_regrets = np.array([np.sum(simutaneous_regrets[:i]) for i in range(1, self._n_iters + 2)])
				

				simutaneous_violation = np.maximum(all_constraints, 0)
				simutaneous_violation = np.sum(simutaneous_violation, axis=1)
				cummulative_violation = np.array([np.sum(simutaneous_violation[:i]) for i in range(1, self._n_iters + 2)])
				
				positive_regret_plus_violation = simutaneous_regrets + simutaneous_violation
				best_positive_regret_plus_violation = [np.min(positive_regret_plus_violation[:i]) for i in range(1, self._n_iters + 2)]
				
				if self._log_plot == True:
					best_positive_regret_plus_violation = np.log10(best_positive_regret_plus_violation)
				
				all_cummu_regrets = np.vstack((all_cummu_regrets, cummulative_regrets))
				all_violations = np.vstack((all_violations, cummulative_violation))
				# all_best_regrets = np.vstack((all_best_regrets, best_simutaneous_regrets_so_far))
				
				all_regret_violation_sum = np.vstack((all_regret_violation_sum, best_positive_regret_plus_violation))
				
				# Get values obtained from optimization alg (without initial points)
				


				# Simple regret 
				# Set infeasible observations to np.inf
				# inf = 1e6
				
				# values_by_optim = all_optimal_values[-self._n_iters:]
				# is_infeasible =  [(torch.FloatTensor(v)>0).any() for v in Di['constraint_values'][-self._n_iters:]]
				# values_by_optim[is_infeasible] = float("inf")
				
				# is_feasible =  [(torch.FloatTensor(v)<=0).all() for v in Di['constraint_values']]
				
				# feasible_values = np.array(values_by_optim)[is_feasible]
				# if feasible_values.shape[0] !=0:
					# min_idx = np.argmin(feasible_values)
					# minimal_value = feasible_values[min_idx]
						# 
					# print(pkl_file, f"min feasible value {minimal_value}")
					# 
					# Reset values to the original "all_optimal_values" array
				
				# all_optimal_values[-self._n_iters:] = values_by_optim

				# feasible_min_value = np.minimum.accumulate(values_by_optim)
				# all_feasible_minimum  = np.vstack((all_feasible_minimum, feasible_min_value))
				
			all_regret_stds = np.std(all_cummu_regrets, 0)
			all_regret_means = np.mean(all_cummu_regrets, 0)
			
			all_violations_std = np.std(all_violations, 0)
			all_violations_mean = np.mean(all_violations, 0)


			# all_feasible_simple_regrets_std = np.std(all_feasible_simple_regrets, 0)
			# all_feasible_simple_regrets_mean = np.mean(all_feasible_simple_regrets, 0)
			
			all_regret_violation_sum_std = np.std(all_regret_violation_sum, 0)
			all_regret_violation_sum_mean = np.mean(all_regret_violation_sum, 0)
			
			
			# all_min_std = np.std(all_min_values, 0)
			# all_min_mean = np.mean(all_min_values, 0)
			
			if self._minimize == False:
				all_regret_means = np.array([-v for v in all_regret_means])

			plotting_objs['regret_mean'].append(all_regret_means)
			plotting_objs['regret_std'].append(all_regret_stds)
			
			plotting_objs['violation_mean'].append(all_violations_mean)
			plotting_objs['violation_std'].append(all_violations_std)
			
			# plotting_objs['feasible_minimum_mean'].append(feasible_min_value)
			# plotting_objs['feasible_minimum_std'].append(feasible_min_value)

			plotting_objs['rv_mean'].append(all_regret_violation_sum_mean)
			plotting_objs['rv_std'].append(all_regret_violation_sum_std)
			
			# plotting_objs['min_mean'].append(all_min_mean)
			# plotting_objs['min_std'].append(all_min_std)
			
			plotting_objs['colors'].append(colors_map[alg])
			plotting_objs['markers'].append(markers_map[alg])
			plotting_objs['labels'].append(alg)
			

		
		for metric in ['rv']: # 'regret', 'violation', 
			metric_std = f"{metric}_std"
			metric_mean = f"{metric}_mean"
			for (mean, std, color, label, marker) in zip(plotting_objs[metric_mean], plotting_objs[metric_std], plotting_objs['colors'], 
												plotting_objs['labels'], plotting_objs['markers']):
				# std =  std[:self._n_iters+1]
				# mean = mean[:self._n_iters+1]
				plt.plot(np.arange(from_iter,to_iter), mean[from_iter:to_iter], label=label, color=color, marker=marker, markevery=self._n_iters//25, markersize=3)
				plt.fill_between(np.arange(from_iter,to_iter), (mean - STD_SCALE*std)[from_iter:to_iter], (mean + STD_SCALE*std)[from_iter:to_iter], alpha=0.1, color=color)
			
			
			title = f"{self._func_name.replace('_',' ')} ({self._dim})"
			title = break_camel_case(title)
			
			fig = plt.gcf()
			ax = fig.gca()
			size = fig.get_size_inches()
			desired_aspect_ratio = 4/3
			size[0] = size[1] * desired_aspect_ratio

			fig.set_size_inches(size)
			leg = fig.legend(fontsize=10, bbox_to_anchor=(0.5, -0.07),loc='lower center', ncol=5, markerscale=2)
			for legobj in leg.legend_handles:
				legobj.set_linewidth(3.0)
			if metric == 'regret':
				ylabel = r'Cummulative Positive Regret $R_T^+$'
			elif  metric=='violation':
				ylabel = r'Cummulative Violation $V_T$'
			elif metric == 'rv':
				ylabel = "Best Positive Regret plus Violation"
				if self._log_plot == True:
					ylabel = r'$Log_{10}$' + f"({ylabel})"
			elif metric == 'min':
				ylabel = f"Min values found"
			else:
				ylabel = f"Best Regret"
			
			
			ax.set_title(title, fontsize=14)
			ax.set_ylabel(ylabel, fontsize=14)
			ax.set_xlabel('Number of evaluations', fontsize=14)
			# plt.ylim(bottom=0)
			plt.xlim(left=0, right=self._n_iters)
			
			# ax.yaxis.grid(True)


			fig.tight_layout() 
			if os.path.isdir("figures") ==False:
				os.makedirs("figures")
			fig.savefig(f'figures/{self._func_name}_dim_{self._dim}_iter_{self._n_iters}_{metric}.pdf', dpi=300,  bbox_inches='tight',pad_inches = 0.1)
			fig.savefig(f'figures/{self._func_name}_dim_{self._dim}_iter_{self._n_iters}_{metric}.png', dpi=300,  bbox_inches='tight',pad_inches = 0.1)
			plt.clf()

class PlotMultipleRegret:
	def __init__(self, exp_info, result_dir, list_colors, list_markers, list_algs, n_trials=10, minimize=True, log_plot=False) -> None:
		# self._dim = dim
		# self._n_iters = n_iters_list
		# self._func_name = obj_name_list
		self._minimize = minimize
		self._exp_info = exp_info
		self._log_plot = log_plot
		self._result_dir = result_dir
		self._n_trials = n_trials
		self._colors = list_colors
		self._markers = list_markers
		self._algs = list_algs
	def assign_color_and_marker(self):
		assert (len(self._algs) <= len(self._colors)) and (len(self._algs) <= len(self._markers)), \
		"The numbers of colors or markers must be larger than the number of algorithms"
		color_map = {}
		marker_map = {}
		for i, alg in enumerate(self._algs):
			color_map[alg] = self._colors[i]
			marker_map[alg] = self._markers[i]
		return color_map, marker_map
	def create_single_alg_plotting_obj(self, obj_name, dim, n_iters):
		rdir = os.path.join(self._result_dir, f"{obj_name}_DIM_{dim}_ITERS_{n_iters}/")
		colors_map, markers_map = self.assign_color_and_marker()
		single_alg_plotting_obj = {'regret_mean':[],
				   		 'regret_std':[],
						#  'feasible_minimum_mean':[],
				   		#  'feasible_minimum_std':[], 
						 'violation_mean':[], 
						 'violation_std':[],
						 'rv_mean':[],
						 'rv_std':[],
						#  'min_mean':[],
						#  'min_std':[],
						 'colors': [], 
						 'labels':[], 
						 'markers':[]}
		for alg in self._algs: 	
			print(alg)
			res_files = glob.glob(f'{rdir}/{alg}/*')
			all_cummu_regrets = np.empty((0, n_iters+1))
			all_violations =  np.empty((0, n_iters+1))
			all_regret_violation_sum = np.empty((0, n_iters+1))
			for pkl_file in res_files[:self._n_trials]:
				
				Di  = pkl.load(open(pkl_file,'rb'))	
				best_feasible_y = dict(Di['alg_configs'])['best_feasible_y']
				all_optimal_values = np.array(Di['optimal_values'])
				all_constraints = np.array(Di['constraint_values'])
				
				
				simutaneous_regrets = np.maximum(0, all_optimal_values - best_feasible_y)
				# best_simutaneous_regrets_so_far = [np.min(simutaneous_regrets[:i]) for i in range(1, self._n_iters + 2)]
				cummulative_regrets = np.array([np.sum(simutaneous_regrets[:i]) for i in range(1, n_iters + 2)])
				

				simutaneous_violation = np.maximum(all_constraints, 0)
				simutaneous_violation = np.sum(simutaneous_violation, axis=1)
				cummulative_violation = np.array([np.sum(simutaneous_violation[:i]) for i in range(1, n_iters + 2)])
				
				positive_regret_plus_violation = simutaneous_regrets + simutaneous_violation
				best_positive_regret_plus_violation = [np.min(positive_regret_plus_violation[:i]) for i in range(1, n_iters + 2)]
				
				if self._log_plot == True:
					best_positive_regret_plus_violation = np.log10(best_positive_regret_plus_violation)
				
				all_cummu_regrets = np.vstack((all_cummu_regrets, cummulative_regrets))
				all_violations = np.vstack((all_violations, cummulative_violation))
				all_regret_violation_sum = np.vstack((all_regret_violation_sum, best_positive_regret_plus_violation))
				

			all_regret_stds = np.std(all_cummu_regrets, 0)
			all_regret_means = np.mean(all_cummu_regrets, 0)
			
			all_violations_std = np.std(all_violations, 0)
			all_violations_mean = np.mean(all_violations, 0)
			
			all_regret_violation_sum_std = np.std(all_regret_violation_sum, 0)
			all_regret_violation_sum_mean = np.mean(all_regret_violation_sum, 0)
			
			if self._minimize == False:
				all_regret_means = np.array([-v for v in all_regret_means])

			single_alg_plotting_obj['regret_mean'].append(all_regret_means)
			single_alg_plotting_obj['regret_std'].append(all_regret_stds)
			
			single_alg_plotting_obj['violation_mean'].append(all_violations_mean)
			single_alg_plotting_obj['violation_std'].append(all_violations_std)

			single_alg_plotting_obj['rv_mean'].append(all_regret_violation_sum_mean)
			single_alg_plotting_obj['rv_std'].append(all_regret_violation_sum_std)

			single_alg_plotting_obj['colors'].append(colors_map[alg])
			single_alg_plotting_obj['markers'].append(markers_map[alg])
			single_alg_plotting_obj['labels'].append(alg)
		return single_alg_plotting_obj
	def plot_multiple_regret(self, objective_func_list):
		nrows = 2
		ncols = 3
		figsize = (12,8)
		nfigs = len(objective_func_list)

		if nfigs % ncols !=0:
			fig, axs = subplots_centered(nrows, ncols, figsize, nfigs)
		else:
			fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
			axs = axs.flatten()
		
		for kf, obj_name in enumerate(objective_func_list):
			dim = self._exp_info[obj_name]['DIM']
			n_iters = self._exp_info[obj_name]['N_ITERS']
			from_iter=0
			to_iter=n_iters

			plotting_objs = self.create_single_alg_plotting_obj(obj_name, dim, n_iters)

			for metric in ['rv']: # 'regret', 'violation', 
				metric_std = f"{metric}_std"
				metric_mean = f"{metric}_mean"
				for (mean, std, color, label, marker) in zip(plotting_objs[metric_mean], plotting_objs[metric_std], plotting_objs['colors'], 
													plotting_objs['labels'], plotting_objs['markers']):
					# std =  std[:RNUM+1]
					# mean = mean[:RNUM+1]
					mark_every = n_iters//25
					
					axs[kf].plot(np.arange(from_iter,to_iter), mean[from_iter:to_iter], label=label, color=color, marker=marker, markevery=mark_every, markersize=3)
					axs[kf].fill_between(np.arange(from_iter,to_iter), (mean - STD_SCALE*std)[from_iter:to_iter], (mean + STD_SCALE*std)[from_iter:to_iter], alpha=0.1, color=color)
			
			title = break_camel_case(f"{obj_name} ({dim})")
			axs[kf].set_title(title, fontsize=12)
		
		

		# Dash line bound for 2x3 figure

		# Get the bounds of the left group (top-left and bottom-left subplots)
		top_left_bounds = axs[0].get_position()
		bottom_left_bounds = axs[4].get_position()  # Bottom-right of the left group

		# Calculate the rectangle parameters for the left group
		x_left_leftgroup = top_left_bounds.x0 - 0.08  # Slightly to the left of the left group
		y_bottom_leftgroup = bottom_left_bounds.y0 - 0.1 # Bottom of the lower left subplot
		rect_width_leftgroup = bottom_left_bounds.x1 - x_left_leftgroup  + 0.05  # Width of the left group
		rect_height_leftgroup = top_left_bounds.y1 - y_bottom_leftgroup + 0.12 # Height of the left group

		# Create a dashed rectangle around the left group
		rect_leftgroup = patches.Rectangle(
			(x_left_leftgroup, y_bottom_leftgroup), rect_width_leftgroup, rect_height_leftgroup, 
			linestyle='--', edgecolor='black', linewidth=1.2, fill=False
		)

		# Add the rectangle for the left group
		fig.add_artist(rect_leftgroup)
		
		fig.text(0.37, 1.02, 'Synthetic benchmark functions', fontsize=14, ha='center', weight='bold')
		fig.text(0.845, 1.02, 'Real-world use cases', fontsize=14, ha='center', weight='bold')

		top_right_bounds = axs[2].get_position()
		bottom_right_bounds = axs[5].get_position()

		# Calculate the rectangle parameters
		x_left = top_right_bounds.x0 + 0.01  # Slightly to the left of the right group
		y_bottom = bottom_right_bounds.y0 - 0.1  # Bottom of the lower right subplot
		rect_width = top_right_bounds.x1 - x_left + 0.1  # Width of the right group
		rect_height = top_right_bounds.y1 - y_bottom + 0.12  # Height from bottom to top of right group

		# Create a dashed rectangle around the right group
		rect = patches.Rectangle(
			(x_left, y_bottom), rect_width, rect_height, 
			linestyle='--', edgecolor='black', linewidth=1.2, fill=False
		)

		# Add the rectangle to the figure
		fig.add_artist(rect)



		if metric == 'regret':
				ylabel = r'Cummulative Positive Regret $R_T^+$'
		elif metric=='violation':
			ylabel = r'Cummulative Violation $V_T$'
		elif metric == 'rv':
			ylabel = "Best Positive Regret plus Violation"
			if self._log_plot == True:
				ylabel = r'$Log_{10}$' + f"({ylabel})"
		elif metric == 'min':
			ylabel = f"Min values found"
		else:
			ylabel = f"Best Regret"
				
		for i, _ in enumerate(range(nfigs)):
			if i%ncols==0:
				axs[i].set_ylabel(ylabel, fontsize=12, labelpad=20)
			axs[i].set_xlabel('Number of evaluations', fontsize=12)
			axs[i].grid()

			handles, labels = axs[i].get_legend_handles_labels()
		
		
		leg = fig.legend(handles, labels, fontsize=12, bbox_to_anchor=(0.5, -0.07),loc='lower center', ncol=5)
		for legobj in leg.legend_handles:
			legobj.set_linewidth(1.5)
		plt.subplots_adjust(wspace=1, hspace=1)

		filename = '-'.join(objective_func_list)
		fig.tight_layout()
		plt.savefig(f'figures/{filename}.pdf', dpi=300,  bbox_inches='tight',pad_inches = 0.1)
		plt.savefig(f'figures/{filename}.png', dpi=300,  bbox_inches='tight',pad_inches = 0.1)
		plt.clf()



class PlotSensitiveSample:
	def __init__(self, exp_info, result_dir, list_colors, list_markers, list_algs, n_trials=10, minimize=True, log_plot=False) -> None:
		# self._dim = dim
		# self._n_iters = n_iters_list
		# self._func_name = obj_name_list
		self.obj_name = 'SensitiveSample'
		self._minimize = minimize
		self._exp_info = exp_info
		self._log_plot = log_plot
		self._result_dir = result_dir
		self._n_trials = n_trials
		self._colors = list_colors
		self._markers = list_markers
		self._algs = list_algs
		self._n_tamped_models = 1000
		self.Ns = range(1,11)
	def assign_color_and_marker(self):
		assert (len(self._algs) <= len(self._colors)) and (len(self._algs) <= len(self._markers)), \
		"The numbers of colors or markers must be larger than the number of algorithms"
		color_map = {}
		marker_map = {}
		for i, alg in enumerate(self._algs):
			color_map[alg] = self._colors[i]
			marker_map[alg] = self._markers[i]
		return color_map, marker_map
	def evaluate_detection(self):
		detection_rates = {alg: [0]*len(self.Ns) for alg in self._algs} 
		
		device = torch.device(0 if torch.cuda.is_available() else 'cpu')
		model = MLP_MNIST(dropout=0.0, num_classes=10, n_hidden=10).to(device)
		model.load_state_dict(torch.load("models_and_data/pretrained_models/trained_MNIST_acc_0.93.pth"))
		model.eval()
		for j in tqdm(range(self._n_tamped_models)):
			tamped_model_path = 'models_and_data/pretrained_models/tamped_models/tamped_mnist.{j}.pth'
			if os.path.isfile(f'')==False:
				tamped_model = copy.deepcopy(model)
				with torch.no_grad():
					for i in range(tamped_model.layer1.weight.shape[0]):
							sz = tamped_model.layer1.weight[i].shape[0]
							tamped_model.layer1.weight[i] += torch.normal(torch.zeros(sz), torch.full([sz], 0.04)).to(device)
						
					for i in range(tamped_model.layer2.weight.shape[0]):
							sz = tamped_model.layer2.weight[i].shape[0]
							tamped_model.layer2.weight[i] += torch.normal(torch.zeros(sz), torch.full([sz], 0.04)).to(device)
				tamped_model.eval()
				torch.save(tamped_model.state_dict(), tamped_model_path)
			else:
				tamped_model = MLP_MNIST(dropout=0.0, num_classes=10, n_hidden=10).to(device)
				tamped_model.load_state_dict(torch.load(tamped_model_path))
			n_iters = self._exp_info[self.obj_name]['N_ITERS']
			for n in self.Ns:		
				for alg in self._algs:
					images = glob.glob(f'asset/sensitive_sample_best/{alg}_{n_iters}/*')
					images = random.choices(images, k=n)
					
					for img_name in images:
						img = cv2.imread(img_name,0)
						img = torch.tensor(img).unsqueeze(1).float().div(255).sub_(0.1307).div_(0.3081).to(device)

						if torch.argmax(tamped_model(img)) != torch.argmax(model(img)):
							if alg in detection_rates:
								detection_rates[alg][n-1]+=1
								break
			
		return detection_rates
	def plot_detection_rates(self):
		# detection_rates = self.evaluate_detection()
		detection_rates = {'Neural-CBO': [228, 382, 493, 585, 620, 695, 739, 777, 802, 832], 
					 'ConfigOpt': [138, 258, 326, 410, 493, 530, 577, 600, 627, 650], 
					 'cEI': [177, 292, 365, 445, 533, 554, 609, 645, 668, 688], 
					 'UCBC': [228, 361, 482, 541, 622, 664, 719, 728, 765, 779], 
					 'ADMMBO': [177, 330, 413, 491, 542, 598, 657, 700, 708, 737]}
		print(detection_rates)
		colors_map, markers_map = self.assign_color_and_marker()
		for alg in self._algs:
			acc = detection_rates[alg]
			plt.plot(self.Ns, np.array(acc)/self._n_tamped_models, label = alg, color = colors_map[alg],  marker=markers_map[alg])

		plt.title(f"Sensitive Sample Detection Rates", fontsize=12)
		fig = plt.gcf()
		leg = plt.legend(fontsize=9, bbox_to_anchor=(0.5, -0.25),loc='lower center', ncol=5)
		for legobj in leg.legend_handles:
			legobj.set_linewidth(3.0)
		plt.ylabel("Detection Rates", fontsize=12)
		plt.xlabel("Number of Samples", fontsize=12)
		plt.grid()

		# fig.tight_layout() 
		if os.path.isdir("figures") ==False:
			os.makedirs("figures")
		# plt.savefig(f'figures/sensitive_sample_detection_rate.pdf', dpi=300,  bbox_inches='tight',pad_inches = 0.1)
		plt.savefig(f'figures/sensitive_sample_detection_rate.png', dpi=300,  bbox_inches='tight',pad_inches = 0.1)
		# plt.savefig('figures/sensitive_sample_detection_rate.png')
		plt.clf()
 

if __name__ == '__main__':
	
	# Plot single regrets

	algs = ['Neural-CBO', 'ConfigOpt', 'cEI', 'UCBC', 'ADMMBO'] 	
	TARGET_FUNCS = ['Branin', 'Simionescu', "GasTransmission", 'Ackley', 'Hartmann', "SpeedReducer"]
	for i, F_NAME in enumerate(TARGET_FUNCS):
		print(F_NAME)
		info = TARGET_INFO[F_NAME]
		plotting_obj = PlotSingleRegret(dim=info['DIM'], n_iters=info['N_ITERS'], n_trials=info['N_TRIALS'], func_name=F_NAME, result_dir='results/',
							list_colors=FIXED_COLORS, list_markers=FIXED_MARKERS,
							list_algs=algs, log_plot=True)
		plotting_obj.plot_single_regret(from_iter=0, to_iter=info['N_ITERS'])
	
	# Plot multiple regrets
	regret_obj = PlotMultipleRegret(exp_info=TARGET_INFO, result_dir='results/',
							list_colors=FIXED_COLORS, list_markers=FIXED_MARKERS,
							list_algs=algs, log_plot=True)
	regret_obj.plot_multiple_regret(TARGET_FUNCS)
	

	# plot_obj = PlotSensitiveSample(exp_info=TARGET_INFO, result_dir='results/',
	# 						list_colors=FIXED_COLORS, list_markers=FIXED_MARKERS,
	# 						list_algs=algs, log_plot=True)
	
	

	