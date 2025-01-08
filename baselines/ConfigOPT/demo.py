import numpy as np
import config
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from copy import copy
from scipy.stats import norm
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

problem_name = 'sinusodal'

optimization_config = {
    'eval_budget': 100
}
base_opt_config = {
    'noise_level':0.0,
    'kernel_var':0.1,
    'train_noise_level': 0.0,
}

def get_optimizer(optimizer_type, optimizer_config, problem_config):
    problem = config.OptimizationProblem(problem_config)
    if optimizer_type == 'safe_bo':
        opt = config.SafeBO(problem, optimizer_config)
        best_obj_list = [-opt.best_obj]
    if optimizer_type == 'constrained_bo':
        opt = config.ConstrainedEI(problem, optimizer_config)
        best_obj_list = [opt.best_obj]
    # if optimizer_type == 'pdbo':
    #     opt = pdbo.PDBO(problem, optimizer_config)
    #     best_obj_list = [opt.best_obj]
    if optimizer_type == 'config':
        opt = config.CONFIGOpt(problem, optimizer_config)
        best_obj_list = [opt.best_obj]
    total_cost_list = []
    return opt, best_obj_list, total_cost_list

problem_config = config.get_config(problem_name)
obj_vals_arr = problem_config['obj'](problem_config['parameter_set'])
constr_vals_arr = problem_config['constrs_list'][0](problem_config['parameter_set'])


# Initialize the config optimizer
configOpt_config = base_opt_config.copy()
configOpt_config.update({
        'total_eval_num': 100,
    }
)
config_opt, configOpt_best_obj_list, configOpt_total_cost_list = get_optimizer(
            'config', configOpt_config, problem_config)
config_opt_obj_list = []
config_opt_constr_list = []

# Initialize SafeOPT
SafeOpt_config = base_opt_config.copy()
SafeOpt_config.update({
        'total_eval_num': 100,
    }
)
safe_opt, SafeOpt_best_obj_list, SafeOpt_total_cost_list = get_optimizer(
            'safe_bo', SafeOpt_config, problem_config)
safe_opt_obj_list = []
safe_opt_constr_list = []

# Initialize CEI
CEIOpt_config = base_opt_config.copy()
CEIOpt_config.update({
        'total_eval_num': 100,
    }
)
CEI_opt, CEI_Opt_best_obj_list, CEIOpt_total_cost_list = get_optimizer(
            'constrained_bo', CEIOpt_config, problem_config)
cei_opt_obj_list = []
cei_opt_constr_list = []

cumu_pos_subopt_config = []
cumu_pos_vio_config = []

cumu_pos_subopt_safe = []
cumu_pos_vio_safe = []

from scripts.fig_hp import *
WIDTH = WIDTH * 1.5
HEIGHT = HEIGHT * 1.5 



def get_con_regret(obj_list, constr_list):
    obj_arr = np.array(obj_list)
    constr_arr = np.array(constr_list)
    # pos_regret_arr = np.maximum(obj_arr-problem_config['f_min'],0)
    min_value_found_so_far = [np.min(obj_arr[:i]) for i in range(1, len(obj_arr))]
    pos_constr_arr = np.maximum(constr_arr, 0)
    all_constr_arr = np.sum(pos_constr_arr, axis=1)
    return min_value_found_so_far, np.minimum.accumulate(all_constr_arr)


def gather_results(opt, obj_list, constr_list):
    y_obj, constr_vals = opt.make_step()
    obj_list.append(y_obj[0, 0])
    constr_list.append([constr_vals[0, k] for k in range(problem_config['num_constrs'])])
    return obj_list, constr_list
    
if __name__ == '__main__':
    Run_steps = 20
    for _ in range(Run_steps):
        config_opt_obj_list, config_opt_constr_list = gather_results(config_opt,config_opt_obj_list,config_opt_constr_list)
        safe_opt_obj_list, safe_opt_constr_list = gather_results(safe_opt,safe_opt_obj_list, safe_opt_constr_list)
        cei_opt_obj_list, cei_opt_constr_list = gather_results(CEI_opt, cei_opt_obj_list, cei_opt_constr_list)
        
        
        
    config_min_value_found_so_far, config_constr_regret = get_con_regret(config_opt_obj_list, config_opt_constr_list)

    safe_min_value_found_so_far, safe_constr_regret = get_con_regret(safe_opt_obj_list, safe_opt_constr_list)

    cei_min_value_found_so_far, cei_constr_regret = get_con_regret(cei_opt_obj_list, cei_opt_constr_list)
    
    plt.plot(config_constr_regret, label='CONFIG')
    plt.plot(safe_constr_regret, label='SafeOpt')
    plt.plot(cei_constr_regret, label='cEI')
    plt.legend()
    plt.savefig('regret.png')
    plt.clf()
    

    plt.plot(config_min_value_found_so_far, label='CONFIG')
    plt.plot(safe_min_value_found_so_far, label='SafeOpt')
    plt.plot(cei_min_value_found_so_far, label='cEI')
    
    plt.savefig('values.png')