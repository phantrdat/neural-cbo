import torch

from helpers import expected_improvement, gaussian_cdf


class EIC:
    def __init__(self):
        self.estimator_idx = None
        self.estimator = None
        self.adaptive = False

    @staticmethod
    def expected_improvement_with_constraints(
        max_obs,
        target_posterior_mean,
        target_posterior_std,
        constraint_mean_f_list,
        constraint_std_f_list,
        threshold_list,
    ):
        ei_vals = expected_improvement(
            max_obs, target_posterior_mean, target_posterior_std
        )

        eic_vals = ei_vals
        for i, threshold in enumerate(threshold_list):
            constraint_mean_f = constraint_mean_f_list[i]
            constraint_std_f = constraint_std_f_list[i]

            eic_vals *= 1.0 - gaussian_cdf(
                threshold,
                constraint_mean_f,
                constraint_std_f,
            )

        return eic_vals

    @staticmethod
    def get_least_violated_input(
        target_y, constraint_y_dict, constraint_info_list, cidx_to_moidx, threshold_list
    ):
        zero_arr = torch.zeros_like(target_y)
        violation = torch.zeros_like(target_y)

        for cidx, constraint_info in enumerate(constraint_info_list):
            moidx = cidx_to_moidx[cidx]
            constraint_y = constraint_y_dict[moidx]
            threshold = threshold_list[cidx]

            if constraint_info["inequality"] == "less_than_equal_to":
                violation = violation + torch.max(zero_arr, threshold + constraint_y)
            else:
                violation = violation + torch.max(zero_arr, threshold - constraint_y)

        return torch.argmin(violation)

    @staticmethod
    def get_best_feasible_observed_input(
        target_y, constraint_y_dict, constraint_info_list, cidx_to_moidx, threshold_list
    ):
        feasible_cond = torch.ones_like(target_y, dtype=torch.bool)

        for cidx, constraint_info in enumerate(constraint_info_list):
            moidx = cidx_to_moidx[cidx]
            constraint_y = constraint_y_dict[moidx]
            threshold = threshold_list[cidx]

            if constraint_info["inequality"] == "less_than_equal_to":
                feasible_cond = torch.logical_and(
                    feasible_cond, -constraint_y >= threshold
                )
            else:
                feasible_cond = torch.logical_and(
                    feasible_cond, constraint_y >= threshold
                )

        feasible_idxs = feasible_cond.nonzero()
        if len(feasible_idxs):
            idx = torch.argmax(target_y[feasible_idxs])


            return idx
        return None

    def get_query(
        self,
        threshold_list,
        target,
        constraint,
        other_info,
    ):
        target_mean_f = target["mean_f"]
        target_std_f = target["std_f"]
        constraint_mean_f_list = constraint["mean_f_list"]
        constraint_std_f_list = constraint["std_f_list"]

        constraint_info_list = other_info["constraint_info_list"]
        target_xz = other_info["target_xz"]
        target_y = other_info["target_y"]
        constraint_y_dict = other_info["constraint_y_dict"]
        cidx_to_moidx = other_info["cidx_to_moidx"]

        best_observed_target_y_idx = EIC.get_best_feasible_observed_input(
            target_y,
            constraint_y_dict,
            constraint_info_list,
            cidx_to_moidx,
            threshold_list,
        )

        if best_observed_target_y_idx is None:
            print(
                "There are not any feasible observed inputs! Set best_observed_target_y to the maximum observed target_y."
            )
            best_observed_target_y_idx = torch.argmax(target_y)
            best_observed_target_y = target_y[best_observed_target_y_idx]
            self.estimator = target_xz[best_observed_target_y_idx]

        else:
            best_observed_target_y = target_y[best_observed_target_y_idx]
            self.estimator = target_xz[best_observed_target_y_idx]

        eic_vals = EIC.expected_improvement_with_constraints(
            best_observed_target_y,
            target_mean_f,
            target_std_f,
            constraint_mean_f_list,
            constraint_std_f_list,
            threshold_list,
        )

        query_idx = torch.argmax(eic_vals)
        query_type = [0] + list(
            range(1, len(threshold_list) + 1)
        )
        method = "EIC"
        return (
            query_idx,
            query_type,
            method,
            {"lower_margin_list": None, "upper_margin_list": None},
        )
