import sys
import os
import numpy as np
import torch

import gpytorch
import pickle

from model import GP

import helpers
from helpers import (
    transform_prediction_and_threshold,
    get_regret,
)

from admmbo import ADMMBO
from cmes_ibo import CMES_IBO


class MultiConstraintADMMBO(object):
    def __init__(self, filename_prefix=None):
        self.target_model = None
        self.constraint_model_dict = None
        self.beta_t = None
        self.xz_domain = None
        self.regrets = None

        if filename_prefix is not None:
            self.load(filename_prefix)

    @staticmethod
    def get_upper_lower_bounds(gp_model, input_domain, beta):
        with torch.no_grad():
            preds = GP.predict_f(gp_model, input_domain)
            means = preds.mean
            stds = torch.sqrt(preds.variance)

            upper = means + beta * stds
            lower = means - beta * stds

        return means, stds, lower, upper

    @staticmethod
    def get_constraint_idx_to_model_obs_idx(constraint_info_list):
        cidx_to_moidx = dict(
            zip(
                list(range(len(constraint_info_list))),
                list(range(len(constraint_info_list))),
            )
        )

        for i, constraint_info in enumerate(constraint_info_list):
            if (
                "share_observation_with_constraint" in constraint_info
                and constraint_info["share_observation_with_constraint"] > 0
            ):
                cidx_to_moidx[constraint_info["share_observation_with_constraint"]] = i

        return cidx_to_moidx

    def run(
        self,
        xz_domain,
        target_info,
        constraint_info_list,
        n_bo_iter,
        experiment_identifier,
        n_training_iter_gp_hyper=30,
        update_gp_hyper_every_iter=1,
        is_visualize=False,
    ):
        print(f"Run BO with unknown constraints using: ADMMBO")
        n_constraint = len(constraint_info_list)

        self.dim = xz_domain.shape[1]
        self.xz_domain = xz_domain

        self.target_info = target_info
        self.constraint_info_list = constraint_info_list
        self.n_constraint = len(self.constraint_info_list)

        self.xsize = self.target_info["func"].xsize
        self.zsize = self.target_info["func"].zsize

        self.cidx_to_moidx = MultiConstraintADMMBO.get_constraint_idx_to_model_obs_idx(
            constraint_info_list
        )

        self.target_xz = self.target_info["init_xz"]
        self.target_y = self.target_info["init_y"]

        self.constraint_xz_dict = {}
        self.constraint_y_dict = {}

        for i in range(self.n_constraint):
            self.constraint_xz_dict[self.cidx_to_moidx[i]] = self.constraint_info_list[
                self.cidx_to_moidx[i]
            ]["init_xz"]
            self.constraint_y_dict[self.cidx_to_moidx[i]] = self.constraint_info_list[
                self.cidx_to_moidx[i]
            ]["init_y"]

        self.betas = []
        self.query_types = []

        self.constraint_approx_margins = []

        instantaneous_regrets = [None] * n_bo_iter
        instantaneous_target_regrets = [None] * n_bo_iter
        instantaneous_constraint_regrets = [None] * n_bo_iter
        
        target_values_found = [None] * n_bo_iter
        queries = [None] * n_bo_iter

        estimator_regrets = [None] * n_bo_iter
        estimator_target_regrets = [None] * n_bo_iter
        estimator_constraint_regrets = [None] * n_bo_iter
        
        constrained_values_found = [None] * n_bo_iter

        target_func_eval = self.target_info["func"].get_noiseless_observation(
            self.xz_domain
        )

        constraint_func_eval_dict = {}
        for i in range(self.n_constraint):
            constraint_func_eval_dict[
                self.cidx_to_moidx[i]
            ] = self.constraint_info_list[i]["func"].get_noiseless_observation(
                self.xz_domain
            )

        inequality_type_list = [
            constraint_info["inequality"]
            for constraint_info in self.constraint_info_list
        ]

        thresholds = [
            constraint_info["threshold"]
            for constraint_info in self.constraint_info_list
        ]

        threshold_list = [
            (
                -thresholds[i]
                if inequality_type_list[i] == "less_than_equal_to"
                else thresholds[i]
            )
            for i in range(len(thresholds))
        ]

        max_func_eval_idx = MultiConstraintADMMBO.get_max_constraint_func_eval_idx(
            target_func_eval,
            constraint_func_eval_dict,
            threshold_list,
            inequality_type_list,
        ).squeeze()
        max_func_eval = target_func_eval[max_func_eval_idx]
        maximizer = self.xz_domain[max_func_eval_idx].numpy().squeeze()
        print(f"Optimal value: {max_func_eval.squeeze()} at {maximizer}")

        if is_visualize:
            helpers.visualize_groundtruth(
                self.xsize,
                self.zsize,
                self.target_xz,
                self.constraint_xz_dict,
                target_func_eval,
                constraint_func_eval_dict,
                thresholds,
                maximizer,
                filename=f"asset/{experiment_identifier}_admmbo_groundtruth.pdf",
            )

        self.target_model = helpers.build_gp(self.target_info)

        self.constraint_model_dict = {}
        for i in range(self.n_constraint):
            self.constraint_model_dict[self.cidx_to_moidx[i]] = helpers.build_gp(
                constraint_info_list[self.cidx_to_moidx[i]]
            )

        self.target_model = helpers.retrain_gp(
            self.target_model, self.target_xz, self.target_y, self.target_info
        )

        for i in self.constraint_model_dict:
            self.constraint_model_dict[i] = helpers.retrain_gp(
                self.constraint_model_dict[i],
                self.constraint_xz_dict[i],
                self.constraint_y_dict[i],
                self.constraint_info_list[i],
            )

        acquisition_function = ADMMBO()
        acquisition_function.preload(
            M=1000.0,
            epsilon=0.1,
            delta=0.01,
            xz_domain=self.xz_domain,
            threshold_list=threshold_list,
        )

        self.xt = self.xz_domain[
            torch.randint(low=0, high=xz_domain.shape[0], size=(1,))
        ]
        self.zts = self.xz_domain.clone().repeat(n_constraint, 1)
        self.yts = torch.zeros_like(self.zts)

        get_target_n_query = lambda t: 1
        get_constraint_n_query = lambda t: 1

        inc_tau = 2.0
        dec_tau = 2.0
        rho_mu = 10.0

        def get_rho(s, r, prev_rho):
            norm_r = torch.norm(r)
            norm_s = torch.norm(s)
            if norm_r > rho_mu * norm_s:
                return prev_rho * inc_tau
            elif norm_s > rho_mu * norm_r:
                return prev_rho / dec_tau
            return prev_rho

        rho = 5.0

        t = -1
        # while sum([len(qt) for qt in self.query_types]) < n_bo_iter:
        for _ in range(n_bo_iter):
            t += 1
            no_of_queries = sum([len(qt) for qt in self.query_types])

            print(f"\nIteration {t}, No. of queries {no_of_queries}:")
            self.betas.append(self.target_info["get_beta_t"](t))
            print(f"beta_t: {self.betas[-1]}")

            method = "Unspecified"

            (
                target_mean_f,
                target_std_f,
                target_lower_f,
                target_upper_f,
            ) = MultiConstraintADMMBO.get_upper_lower_bounds(
                self.target_model, self.xz_domain, self.betas[-1]
            )

            constraint_mean_f_list = []
            constraint_std_f_list = []
            constraint_lower_f_list = []
            constraint_upper_f_list = []

            for i in range(self.n_constraint):
                (
                    constraint_mean_f,
                    constraint_std_f,
                    constraint_lower_f,
                    constraint_upper_f,
                ) = MultiConstraintADMMBO.get_upper_lower_bounds(
                    self.constraint_model_dict[self.cidx_to_moidx[i]],
                    self.xz_domain,
                    self.betas[-1],
                )

                if constraint_info_list[i]["inequality"] == "less_than_equal_to":
                    constraint_mean_f_list.append(-constraint_mean_f)
                    constraint_std_f_list.append(constraint_std_f)
                    constraint_lower_f_list.append(-constraint_upper_f)
                    constraint_upper_f_list.append(-constraint_lower_f)
                elif constraint_info_list[i]["inequality"] == "greater_than_equal_to":
                    constraint_mean_f_list.append(constraint_mean_f)
                    constraint_std_f_list.append(constraint_std_f)
                    constraint_lower_f_list.append(constraint_lower_f)
                    constraint_upper_f_list.append(constraint_upper_f)
                else:
                    raise Exception(
                        f"Unknown inequality type {constraint_info_list[i]['inequality']}"
                    )

            for target_query_idx in range(get_target_n_query(t)):
                query_idx, xt = acquisition_function.get_target_query(
                    rho,
                    self.zts,
                    self.yts,
                    self.target_xz,
                    self.target_y,
                    target_mean_f,
                    target_std_f,
                )

                target_query = self.xz_domain[query_idx : query_idx + 1]

                (
                    self.target_xz,
                    self.target_y,
                    self.target_model,
                    self.constraint_xz_dict,
                    self.constraint_y_dict,
                    self.constraint_model_dict,
                ) = helpers.update_target_and_constraint_posterior(
                    [0],
                    target_query,
                    self.target_info,
                    self.target_xz,
                    self.target_y,
                    self.target_model,
                    self.constraint_info_list,
                    self.cidx_to_moidx,
                    self.constraint_xz_dict,
                    self.constraint_y_dict,
                    self.constraint_model_dict,
                    update_gp_hyperparameters=(
                        (t + 1) % update_gp_hyper_every_iter == 0
                    ),
                    training_iter=n_training_iter_gp_hyper,
                )

            prev_zts = self.zts.clone()
            for ci, threshold in enumerate(threshold_list):
                for constraint_query_idx in range(get_constraint_n_query(t)):
                    (
                        ci_query_idx,
                        self.zts[ci, :],
                    ) = acquisition_function.get_constraint_query(
                        rho,
                        self.xt,
                        self.zts[ci : ci + 1, :],
                        self.yts[ci : ci + 1, :],
                        self.constraint_model_dict[self.cidx_to_moidx[ci]],
                        constraint_mean_f_list[ci],
                        constraint_std_f_list[ci],
                        constraint_info_list[ci]["inequality"],
                        self.constraint_xz_dict[self.cidx_to_moidx[ci]],
                        threshold,
                    )

                    (
                        self.target_xz,
                        self.target_y,
                        self.target_model,
                        self.constraint_xz_dict,
                        self.constraint_y_dict,
                        self.constraint_model_dict,
                    ) = helpers.update_target_and_constraint_posterior(
                        [ci + 1],
                        self.xz_domain[ci_query_idx : ci_query_idx + 1],
                        self.target_info,
                        self.target_xz,
                        self.target_y,
                        self.target_model,
                        self.constraint_info_list,
                        self.cidx_to_moidx,
                        self.constraint_xz_dict,
                        self.constraint_y_dict,
                        self.constraint_model_dict,
                        update_gp_hyperparameters=(
                            (t + 1) % update_gp_hyper_every_iter == 0
                        ),
                        training_iter=n_training_iter_gp_hyper,
                    )
            self.yts = self.yts + rho * (xt - self.zts)

            st = rho * (prev_zts - self.zts)
            rt = xt - self.zts
            rho = get_rho(st, rt, rho)
            
            estimator_idx = CMES_IBO.get_estimator_idx(
                target_mean_f,
                constraint_mean_f_list,
                constraint_std_f_list,
                threshold_list,
            )
            if estimator_idx is not None:
                estimator = self.xz_domain[estimator_idx, :]
            else:
                print("WARNING: Estimator is None!")

            print(f"  Estimator = {estimator}")

            (
                instantaneous_regret,
                instantaneous_target_regret,
                instantaneous_constraint_regret,
            ) = get_regret(
                target_query, max_func_eval, target_info, constraint_info_list
            )
            
            print(
                f"  Instantaneous regret = {instantaneous_regret} = target {instantaneous_target_regret} + constraint {instantaneous_constraint_regret}"
            )
            

            # instantaneous_regrets[t] = instantaneous_regret
            # instantaneous_target_regrets[t] = instantaneous_target_regret
            # instantaneous_constraint_regrets[t] = instantaneous_constraint_regret

            target_values_found[t] = target_info["func"].get_noiseless_observation(target_query)
            constrained_values_found[t] = [constraint_info["func"].get_noiseless_observation(target_query) for constraint_info in constraint_info_list]
            queries[t] = target_query.squeeze(0)
            # if estimator is not None:
            #     (
            #         estimator_regret,
            #         estimator_target_regret,
            #         estimator_constraint_regret,
            #     ) = get_regret(
            #         estimator, max_func_eval, target_info, constraint_info_list
            #     )

            #     estimator_regrets[t] = estimator_regret
            #     estimator_target_regrets[t] = estimator_target_regret
            #     estimator_constraint_regrets[t] = estimator_constraint_regret

            #     print(
            #         f"  estimator regret = {estimator_regrets[t]} = target {estimator_target_regrets[t]} + constraint {estimator_constraint_regrets[t]}"
            #     )

                
                 
            self.query_types.append([0] + list(range(1, len(threshold_list) + 1)))
            other_info = {"lower_margin_list": None, "upper_margin_list": None}
            method = "ADMMBO"
            if is_visualize and t >= n_bo_iter - 1:
                helpers.visualize_bo_iteration(
                    self.xsize,
                    self.zsize,
                    self.target_xz,
                    self.constraint_xz_dict,
                    target_upper_f,
                    thresholds,
                    other_info["lower_margin_list"],
                    other_info["upper_margin_list"],
                    constraint_func_eval_dict,
                    constraint_upper_f_list,
                    constraint_lower_f_list,
                    inequality_type_list,
                    self.query_types[t],
                    maximizer,
                    t,
                    method,
                    estimator_regrets[t],
                    target_query.numpy().squeeze(),
                    estimator,
                    filename=f"asset/{experiment_identifier}_admmbo_bo_iter_{t}.pdf",
                )
            sys.stdout.flush()
        self.regrets = {
            "instantaneous_regrets": instantaneous_regrets,
            "instantaneous_target_regrets": instantaneous_target_regrets,
            "instantaneous_constraint_regrets": instantaneous_constraint_regrets,
            "estimator_regrets": estimator_regrets,
            "estimator_target_regrets": estimator_target_regrets,
            "estimator_constraint_regrets": estimator_constraint_regrets,
        }
        queries = torch.stack(queries)
        return self.regrets, self.query_types, target_values_found, constrained_values_found, queries

    @staticmethod
    def get_max_constraint_func_eval(
        func_eval, constraint_eval_dict, threshold_list, inequality_type_list
    ):
        with torch.no_grad():
            idx = MultiConstraintADMMBO.get_max_constraint_func_eval_idx(
                func_eval, constraint_eval_dict, threshold_list, inequality_type_list
            )
            return func_eval[idx]

    @staticmethod
    def get_max_constraint_func_eval_idx(
        func_eval, constraint_eval_dict, threshold_list, inequality_type_list
    ):
        with torch.no_grad():
            feasible_cond = torch.ones_like(func_eval, dtype=torch.bool)

            for i, inequality in enumerate(inequality_type_list):
                constraint_eval = constraint_eval_dict[i]

                feasible_cond = torch.logical_and(
                    feasible_cond,
                    transform_prediction_and_threshold[inequality](constraint_eval)
                    >= threshold_list[i],
                )

            feasible_idxs = feasible_cond.nonzero()
            return feasible_idxs[torch.argmax(func_eval[feasible_idxs])]

    def predict(self, xz):
        xz = xz.reshape(-1, self.dim)
        f_preds = GP.predict_f(self.target_model, xz)

        with torch.no_grad():
            f_means = f_preds.mean
            f_vars = f_preds.variance
            f_stds = torch.sqrt(f_vars)
            lower = f_means - self.beta_t * f_stds
            upper = f_means + self.beta_t * f_stds

        return f_means, f_stds, lower, upper

    def save(self, filename_prefix):
        if self.target_model is None or self.beta_t is None:
            raise Exception("Haven't trained GP model")

        data = {
            "xz_domain": self.xz_domain,
            "dim": self.dim,
            "observed_target_xz": self.observed_target_xz,
            "observed_target_y": self.observed_target_y,
            "beta_t": self.beta_t,
            "has_prior": self.has_prior,
            "use_standardization": self.use_standardization,
            "ard": self.ard,
            "regrets": self.regrets,
        }
        gp_filename = filename_prefix + "__GP.pth"
        self.target_model.save(gp_filename)

        BO_filename = filename_prefix + "__BO_data.pkl"
        with open(BO_filename, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Save BO results to {gp_filename}")
        print(f"               and {BO_filename}")

    def load(self, filename_prefix):
        with open(filename_prefix + "__BO_data.pkl", "rb") as f:
            data = pickle.load(f)
            self.xz_domain = data["xz_domain"]

            self.dim = data["dim"]

            self.observed_target_xz = data["observed_target_xz"]
            self.observed_target_y = data["observed_target_y"]
            self.beta_t = data["beta_t"]
            self.has_prior = data["has_prior"]
            self.use_standardization = data["use_standardization"]
            self.ard = data["ard"]
            self.regrets = data["regrets"] if "regrets" in data else None

            ard_num_dims = self.dim if self.ard else None

        self.target_model = GP(
            self.observed_target_xz, self.observed_target_y, ard=self.ard
        )
        if self.has_prior:
            self.target_model.likelihood.noise_covar.register_prior(
                "noise_std_prior",
                gpytorch.priors.NormalPrior(0.0, 1.0),
                lambda module: module.noise.sqrt(),
            )

            self.target_model.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    lengthscale_prior=gpytorch.priors.GammaPrior(2.0, 4.0),
                    ard_num_dims=ard_num_dims,
                ),
                outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
            )

        GP.load(self.target_model, filename_prefix + "__GP.pth")
