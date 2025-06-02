# Neural-CBO
This repository contains the source code and configurations for our Neural-CBO paper submitted to UAI 2025. 

## Requirements

Make sure you have the following dependencies installed:

- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [BoTorch](https://botorch.org/)

You can install them using `pip`:

```bash
pip install torch numpy botorch

To run the experiment with Neural-CBO with a single GPU, use the following command:
python main.py -cfg <path_to_config> -gpu_id <id>

To run the ConfigOpt/cEI baselines, use the following command:
python run_baselines.py -cfg <path_to_config> -gpu_id <id>

To run the ADMMBO baseline, direct to folder "baselines/UCB-C" and use the following command:
python exec_admmbo.py --path <path_to_config>

To run the UCB-C baseline, direct to folder "baselines/UCB-C" and use the following command:
python exec_bo.py --path <path_to_config>

The config files for ADMMBO/UCB-C is stored in "baselines/UCB-C/expm". The config files for Neural-CBO/ConfigOpt/cEI is stored in "./objective_configs/". 

To plot the results, use file utils/plot_regret.py and follow the example in this file
