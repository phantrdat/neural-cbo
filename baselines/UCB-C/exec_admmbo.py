from pathlib import Path
import sys
import os
import numpy as np
import pickle
import json
import argparse
import time
import torch
from matplotlib import pyplot as plt
from opt_admmbo import MultiConstraintADMMBO
import my_importlib


def standardize(ys):
    mean_y = ys.mean()
    std_y = ys.std()
    return (ys - ys.mean()) / ys.std()


parser = argparse.ArgumentParser()
parser.add_argument("--random", default='1-2', dest="nrand", type=str, help="Number of random runs")
parser.add_argument(
    "--plot", dest="isplot", type=int, help="1 if plotting BO iterations, 0 otherwise"
)
parser.add_argument(
    "--initseed",
    dest="initseed",
    type=int, default=0,
    help="random seed to generate initial observations",
)
parser.add_argument("--path", type=str, default='/home/trongp/constrained_NeuralBO/baselines/UCB-C/expm/hartmann.json', help="Path to experiments config")

args = parser.parse_args()

experiment_config_filename = args.path
experiment_identifier = Path(args.path).stem
nrand = args.nrand if args.nrand else 1
initseed = args.initseed if args.initseed else 0
is_visualize = bool(args.isplot) if args.isplot else False

print(f"Number or random runs: {nrand}")
print(f"Path to config: {experiment_config_filename}")

if Path(experiment_config_filename).suffix == '.pkl':
    with open(experiment_config_filename, "rb") as f:
        data = pickle.load(f)
if Path(experiment_config_filename).suffix == '.json':
    data = json.load(open(experiment_config_filename, "r"))
    
n_bo_iterations = data["n_bo_iterations"]
n_training_iter_gp_hyper = data["n_training_iter_gp_hyper"]
update_gp_hyper_every_iter = data["update_gp_hyper_every_iter"]

xsize = data["xsize"]
zsize = data["zsize"]
domain_size = xsize * zsize
target_get_beta_t = (
    lambda t: 2.0 * np.log(domain_size * (t + 1) ** 2 / 6 / 0.1) / data["beta"]
)
constraint_get_beta_t = (
    lambda t: 2.0 * np.log(domain_size * (t + 1) ** 2 / 6 / 0.1) / data["beta"]
)

target = {
    "name": "target",
    "module_name": data["target"]["module_name"],
    "identifier": data["target"]["identifier"],
    "noise_std": data["target"]["noise_std"],
    "n_init_observations": data["target"]["n_init_observations"],
    "func": None,
    "init_xz": None,
    "init_y": None,
    "is_hyperparameter_trainable": data["target"]["is_hyperparameter_trainable"],
    "gp_prior": data["target"]["gp_prior"],
    "hyperparameters": None,
    "use_standardization": data["target"]["use_standardization"],
    "noise_std": data["target"]["noise_std"],
    "ard": data["target"]["ard"],
    "get_beta_t": target_get_beta_t,
    "fix_mean_at": None,
}

constraint_list = [
    {
        "name": "constraint",
        "share_observation_with_constraint": constraint_info[
            "share_observation_with_constraint"
        ],
        "inequality": constraint_info["inequality"],
        "module_name": constraint_info["module_name"],
        "identifier": constraint_info["identifier"],
        "noise_std": constraint_info["noise_std"],
        "n_init_observations": constraint_info["n_init_observations"],
        "func": None,
        "init_xz": None,
        "init_y": None,
        "standardized_init_y": None,
        "is_hyperparameter_trainable": constraint_info["is_hyperparameter_trainable"],
        "gp_prior": constraint_info["gp_prior"],
        "hyperparameters": None,
        "use_standardization": constraint_info["use_standardization"],
        "noise_std": constraint_info["noise_std"],
        "ard": constraint_info["ard"],
        "threshold": constraint_info["threshold"],
        "get_beta_t": constraint_get_beta_t,
        "fix_mean_at": constraint_info["fix_mean_at"]
        if "fix_mean_at" in constraint_info
        else None,
    }
    for constraint_info in data["constraints"]
]

save_root = f"/home/trongp/constrained_NeuralBO/results/{data['target']['identifier']}_DIM_{data['dimension']}_ITERS_{data['n_bo_iterations']}/ADMMBO"
if os.path.isdir(save_root) ==False:
	os.makedirs(save_root)
# save_result_filename_prefix = f"out/{experiment_identifier}_admmbo"
regrets = {}
query_types = []
nrand = [int(x) for x in nrand.split('-')]
for nr in range(nrand[0], nrand[1]):
    print("##########################")
    print(f"# Random Experiment {nr} #")

    t1 = time.time()
    for i, func_info in enumerate([target] + constraint_list):
        func_info["func"] = my_importlib.importfunction(func_info["module_name"])(
            xsize,
            zsize,
            func_info["identifier"],
            noise_std=func_info["noise_std"],
        )

        if func_info["is_hyperparameter_trainable"]:
            func_info["hyperparameters"] = None
        else:
            func_info["hyperparameters"] = func_info["func"].get_GP_hyperparameters(
                func_info["func"].task_identifier
            )
            func_info["hyperparameters"]["likelihood.noise_covar.noise"] = max(
                0.0001, func_info["noise_std"] ** 2
            )

        func_info["init_xz"], func_info["init_y"] = func_info[
            "func"
        ].get_init_observations(func_info["n_init_observations"], seed= int(t1)%100000)

    xz_domain = target["func"].get_discrete_xz_domain()

    ucb = MultiConstraintADMMBO()
    
    regrets_nr, query_types_nr, target_values_found, constrained_values_found, queries = ucb.run(
        xz_domain,
        target,
        constraint_list,
        n_bo_iterations,
        experiment_identifier + f"_rand_{nr}",
        n_training_iter_gp_hyper=n_training_iter_gp_hyper,
        update_gp_hyper_every_iter=update_gp_hyper_every_iter,
        is_visualize=is_visualize,
    )

    query_types.append(query_types_nr)

    if len(regrets) == 0:
        regrets = regrets_nr
        for key in regrets:
            regrets[key] = np.expand_dims(regrets[key], axis=0)

    else:
        for key in regrets:
            regrets[key] = np.concatenate(
                [regrets[key], np.expand_dims(regrets_nr[key], axis=0)], axis=0
            )

    t2 = time.time() - t1
    sys.stdout.flush()
    
    info = {"alg_configs": data,
			"optimal_values": [-x.item() for x in target_values_found if x != None], # Because we need to plot minimal values, but the baseline run on maximization task 
			"constraint_values": [list(map(torch.Tensor.item, x)) for x in constrained_values_found if x!=None],
            "X_train": queries,
			"Running time": t2}
    

    
    results_filename = f"{save_root}/ADMMBO_{data['target']['identifier']}_dim{data['dimension']}.{nr}.pkl"
    
    with open(results_filename, "wb") as f:
        pickle.dump(info, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Save results to {results_filename}.")

    # regret_filename = f"{save_result_filename_prefix}_regrets.pkl"
    # with open(regret_filename, "wb") as f:
    #     pickle.dump(regrets, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     print(f"Save regrets to {regret_filename}.")

    # bo_info_filename = f"{save_result_filename_prefix}_bo_info.pkl"
    # with open(bo_info_filename, "wb") as f:
    #     pickle.dump(
    #         {
    #             "query_types": query_types,
    #             "target_xz": ucb.target_xz,
    #             "target_y": ucb.target_y,
    #             "constraint_xz_dict": ucb.constraint_xz_dict,
    #             "constraint_y_dict": ucb.constraint_y_dict,
    #             "cidx_to_moidx": ucb.cidx_to_moidx,
    #         },
    #         f,
    #         protocol=pickle.HIGHEST_PROTOCOL,
    #     )
