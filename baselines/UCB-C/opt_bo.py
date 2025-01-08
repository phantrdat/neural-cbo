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
from cmes_ibo import (
    CMES_IBO,
)

ACQUISITION_FUNCTIONS = [
    "ucbc",
    "ucbd",
    "eic",
    "cmes_ibo",
]

from ucbd import UCBD
from ucbc import UCBC
from eic import EIC
from cmes_ibo import CMES_IBO


class MultiConstraintBO(object):
    def __init__(self, filename_prefix=None):
        self.target_model = None
        self.constraint_model_dict = None
        self.beta_t = None
        self.xz_domain = None
        self.regrets = None

        if filename_prefix is not None:
            self.load(filename_prefix)

    @staticmethod
    def get_acquisition_function(acquisition_function_name):
        if acquisition_function_name == "ucbc":
            return UCBC()

        elif acquisition_function_name == "ucbd":
            return UCBD()

        elif acquisition_function_name == "eic":
            return EIC()

        elif acquisition_function_name == "cmes_ibo":
            return CMES_IBO()

        else:
            raise Exception(
                f"Haven't implemented for acquisition function: {acquisition_function_name}!"
            )

        return None

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
        acquisition_function_name,
        n_bo_iter,
        experiment_identifier,
        n_training_iter_gp_hyper=30,
        update_gp_hyper_every_iter=1,
        is_visualize=False,
    ):
        print(f"Run BO with unknown constraints using: {acquisition_function_name}")

        self.dim = xz_domain.shape[1]
        self.xz_domain = xz_domain

        self.target_info = target_info
        self.constraint_info_list = constraint_info_list
        self.n_constraint = len(self.constraint_info_list)

        self.xsize = self.target_info["func"].xsize
        self.zsize = self.target_info["func"].zsize

        self.cidx_to_moidx = MultiConstraintBO.get_constraint_idx_to_model_obs_idx(
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

        max_func_eval_idx = MultiConstraintBO.get_max_constraint_func_eval_idx(
            target_func_eval,
            constraint_func_eval_dict,
            threshold_list,
            inequality_type_list,
        ).squeeze()

        max_func_eval = target_func_eval[max_func_eval_idx]
        maximizer = self.xz_domain[max_func_eval_idx].numpy().squeeze()
        print(f"Optimal value: {max_func_eval.squeeze()} at {maximizer}")

        for i, inequality in enumerate(inequality_type_list):
            constraint_eval = constraint_func_eval_dict[i]
            violation = (
                threshold_list[i]
                - transform_prediction_and_threshold[inequality](constraint_eval)[
                    max_func_eval_idx
                ]
            )
            print("  Constraint violation", i, ":", violation)

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
                filename=f"asset/{experiment_identifier}_{acquisition_function_name}_groundtruth.pdf",
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

        acquisition_function = MultiConstraintBO.get_acquisition_function(
            acquisition_function_name
        )

        t = -1
        # while sum([len(qt) for qt in self.query_types]) < n_bo_iter:
        for _ in range(n_bo_iter):
            t += 1
            no_of_queries = sum([len(qt) for qt in self.query_types])

            print(f"\nIteration {t}, No. of queries {no_of_queries}:")
            self.betas.append(self.target_info["get_beta_t"](t))

            method = "Unspecified"

            (
                target_mean_f,
                target_std_f,
                target_lower_f,
                target_upper_f,
            ) = MultiConstraintBO.get_upper_lower_bounds(
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
                ) = MultiConstraintBO.get_upper_lower_bounds(
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

            (
                query_idx,
                query_type,
                method,
                other_info,
            ) = acquisition_function.get_query(
                threshold_list,
                target={
                    "mean_f": target_mean_f,
                    "std_f": target_std_f,
                    "lower_f": target_lower_f,
                    "upper_f": target_upper_f,
                },
                constraint={
                    "mean_f_list": constraint_mean_f_list,
                    "std_f_list": constraint_std_f_list,
                    "lower_f_list": constraint_lower_f_list,
                    "upper_f_list": constraint_upper_f_list,
                },
                other_info={
                    "constraint_info_list": constraint_info_list,
                    "target_xz": self.target_xz,
                    "target_y": self.target_y,
                    "cidx_to_moidx": self.cidx_to_moidx,
                    "constraint_y_dict": self.constraint_y_dict,
                    "n_sample": 10,
                    "xz_domain": self.xz_domain,
                    "target_model": self.target_model,
                    "constraint_model_dict": self.constraint_model_dict,
                    "constraint_info_list": self.constraint_info_list,
                },
            )
            query = self.xz_domain[query_idx].reshape(1, self.dim)
            self.query_types.append(query_type)

            estimator = None
            if acquisition_function_name == "eic":
                estimator_idx = CMES_IBO.get_estimator_idx(
                    target_mean_f,
                    constraint_mean_f_list,
                    constraint_std_f_list,
                    threshold_list,
                )
                estimator = self.xz_domain[estimator_idx, :].reshape(1, self.dim)

            elif acquisition_function.estimator is not None:
                estimator = acquisition_function.estimator

            elif acquisition_function.estimator_idx is not None:
                estimator = self.xz_domain[acquisition_function.estimator_idx].reshape(
                    1, self.dim
                )
            else:
                print("No estimator provided, use the input query as estimator!")
                estimator = query

            print(f"  Query = {query}")
            queries[t] = query.squeeze(0)
            print(f"  Estimator = {estimator}")
            #######

            print("TARGET Observations:", self.target_xz.shape)
            for i, constraint_xz in self.constraint_xz_dict.items():
                print(f"CONSTRAINT {i} Observations:", constraint_xz.shape)

            (
                self.target_xz,
                self.target_y,
                self.target_model,
                self.constraint_xz_dict,
                self.constraint_y_dict,
                self.constraint_model_dict,
            ) = helpers.update_target_and_constraint_posterior(
                self.query_types[t],
                query,
                self.target_info,
                self.target_xz,
                self.target_y,
                self.target_model,
                self.constraint_info_list,
                self.cidx_to_moidx,
                self.constraint_xz_dict,
                self.constraint_y_dict,
                self.constraint_model_dict,
                update_gp_hyperparameters=((t + 1) % update_gp_hyper_every_iter == 0),
                training_iter=n_training_iter_gp_hyper,
            )

            (
                instantaneous_regret,
                instantaneous_target_regret,
                instantaneous_constraint_regret,
            ) = get_regret(query, max_func_eval, target_info, constraint_info_list)
            
            print(
                f"  Instantaneous regret = {instantaneous_regret} = target {instantaneous_target_regret} + constraint {instantaneous_constraint_regret}"
            )
            instantaneous_regrets[t] = instantaneous_regret
            instantaneous_target_regrets[t] = instantaneous_target_regret
            instantaneous_constraint_regrets[t] = instantaneous_constraint_regret
            
            target_values_found[t] = target_info["func"].get_noiseless_observation(query)
            constrained_values_found[t] = [constraint_info["func"].get_noiseless_observation(query) for constraint_info in constraint_info_list]
            if estimator is not None:
                (
                    estimator_regret,
                    estimator_target_regret,
                    estimator_constraint_regret,
                ) = get_regret(
                    estimator, max_func_eval, target_info, constraint_info_list
                )

                estimator_regrets[t] = estimator_regret
                estimator_target_regrets[t] = estimator_target_regret
                estimator_constraint_regrets[t] = estimator_constraint_regret

                print(
                    f"  estimator regret = {estimator_regrets[t]} = target {estimator_target_regrets[t]} + constraint {estimator_constraint_regrets[t]}"
                )

            if is_visualize and (t == 0 or (t + 1) % 20 == 0):
                helpers.visualize_bo_iteration(
                    self.xsize,
                    self.zsize,
                    self.target_xz,
                    self.constraint_xz_dict,
                    target_upper_f,
                    thresholds,
                    None,
                    None,
                    constraint_func_eval_dict,
                    constraint_upper_f_list,
                    constraint_lower_f_list,
                    inequality_type_list,
                    self.query_types[t],
                    maximizer,
                    t,
                    method,
                    estimator_regrets[t],
                    query.numpy().squeeze(),
                    estimator,
                    filename=f"asset/{experiment_identifier}_{acquisition_function_name}_bo_iter_{t}.pdf",
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
            idx = MultiConstraintBO.get_max_constraint_func_eval_idx(
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
        data = {
            "xz_domain": self.xz_domain,
            "dim": self.dim,
            "target_xz": self.target_xz,
            "target_y": self.target_y,
        }

        BO_filename = filename_prefix + "__BO.pkl"
        with open(BO_filename, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Save BO results to {BO_filename}")
