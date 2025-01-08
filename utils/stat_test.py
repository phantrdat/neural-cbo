from statsmodels.stats.weightstats import ztest, ttest_ind, ztest
from statsmodels.stats.multitest import fdrcorrection
import glob
import numpy as np
import pickle as pkl
import pandas as pd
from scipy.stats import kstest as ks
from scipy import stats
TRIAL = 10
# Ds = [2,2,4, 5,6, 7]
# N_ITERS = [100, 100, 100, 100, 200, 200]
# NAMES = ['BraninHoo', 'Simionescu', 'GasTransmission', 'Ackley', 'Hartmann', 'SpeedReducer']

Ds = [4, 7]
N_ITERS = [100, 100]
NAMES = ['GasTransmission', 'SpeedReducer']

ALGS = ['Neural-CBO', 'ConfigOpt', 'cEI', 'UCBC', 'ADMMBO']
pairs = [(baseline, ALGS[0]) for baseline in ALGS[1:]]



def ks_test(func_name, D, RNUM, alg):
	directory =  f"results/{func_name}_DIM_{D}_ROUNDS_{RNUM}"
	res_files = glob.glob(f'{directory}/{alg}/*')
	def get_optimum(res_files):
		optimum = []
		for pkl_file in res_files[:TRIAL]:
			Di  = pkl.load(open(pkl_file,'rb'))
			optimum_each_run = np.array(Di['optimal_values'])
			optimum_each_run = np.array([np.min(optimum_each_run[:i]) for i in range(1, RNUM + 2)])			
			
			optimum.append(np.min(Di['optimal_values']))
		return optimum
	
	optimum = get_optimum(res_files)
	optimum_1 = optimum
	optimum_1 = (optimum_1 - np.mean(optimum_1))/ np.std(optimum_1)


	p_value = ks(optimum_1, stats.norm.cdf).pvalue
	p_value = float('{:0.2e}'.format(p_value))
	return p_value

def t_test():
	results_dict = {}
	for kf, (func_name, D, n_iters) in enumerate(zip(NAMES , Ds, N_ITERS)):
		pvals = []
		for pair in pairs: 
			if func_name not in results_dict:
				results_dict[func_name] = {}
			
			alg1, alg2 = pair
			directory =  f"results/{func_name}_DIM_{D}_ITERS_{n_iters}"
			res_files_1 = glob.glob(f'{directory}/{alg1}/*')
			res_files_2 = glob.glob(f'{directory}/{alg2}/*')

			
			
			def get_optimum(res_files):
				optimum = []

				for pkl_file in res_files[:TRIAL]:
					Di  = pkl.load(open(pkl_file,'rb'))	
					best_feasible_y = dict(Di['alg_configs'])['best_feasible_y']
					all_optimal_values = np.array(Di['optimal_values'])
					all_constraints = np.array(Di['constraint_values'])
					
					
					simutaneous_regrets = np.maximum(0, all_optimal_values - best_feasible_y)
					# best_simutaneous_regrets_so_far = [np.min(simutaneous_regrets[:i]) for i in range(1, self._n_iters + 2)]
					
					simutaneous_violation = np.maximum(all_constraints, 0)
					simutaneous_violation = np.sum(simutaneous_violation, axis=1)
					
					positive_regret_plus_violation = simutaneous_regrets + simutaneous_violation
					best_positive_regret_plus_violation = [np.min(positive_regret_plus_violation[:i]) for i in range(1, n_iters + 2)]
					best_positive_regret_plus_violation = np.log10(best_positive_regret_plus_violation)
				return best_positive_regret_plus_violation
			

			optimum_1 = get_optimum(res_files_1)
			optimum_2 = get_optimum(res_files_2)
			(_, p_value, _) = ttest_ind(optimum_1, optimum_2, alternative='larger')
			# z_score = float('{:0.2e}'.format(z_score))
			p_value = float('{:0.2e}'.format(p_value))
			pvals.append(p_value)
		
		results_dict[func_name][D] = list(zip(pvals, list(fdrcorrection(pvals)[0])))
	return results_dict

if __name__ =='__main__':
	# results_dict = {}
	# for (func_name,d,n) in zip(NAMES,Ds,RNUMS):
	# 	if func_name not in results_dict:
	# 		results_dict[func_name] = {}
	# 	ks_pvalue = []
	# 	for alg in ALGS:
	# 		pv = ks_test(func_name, d, n, alg)
	# 		if pv <=0.05:
	# 			print(func_name,d,n,alg)
	# 		ks_pvalue.append(pv)
	# 	results_dict[func_name][d] = ks_pvalue
	# df = pd.DataFrame.from_dict(results_dict)
	# df.to_csv('ks.csv')
	print(pairs)
	results_dict = t_test()
	df = pd.DataFrame.from_dict(results_dict)
	df.to_csv('ztest.csv')


