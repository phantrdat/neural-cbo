import torch

from helpers import gaussian_cdf, gaussian_log_pdf, expected_improvement

from model import GP


class ADMMBO:
    def __init__(self):
        self.estimator_idx = None
        self.estimator = None
        self.adaptive = False

    def preload(self, M, epsilon, delta, xz_domain, threshold_list):
        self.xz_domain = xz_domain
        self.threshold_list = threshold_list
        self.M = M
        self.epsilon = epsilon
        self.delta = delta

        self.dim = self.xz_domain.shape[1]

    @staticmethod
    def get_optimality_objective(target_vals, target_xz, zts, yts, rho):
        n_constraint = zts.shape[0]
        optimality_objective = target_vals
        for i in range(n_constraint):
            optimality_objective = optimality_objective - rho / 2.0 * torch.square(
                torch.norm(
                    target_xz - zts[i : i + 1, :] + yts[i : i + 1, :] / rho,
                    dim=1,
                )
            )
        return optimality_objective

    def get_target_query(
        self,
        rho,
        zts,
        yts,
        target_xz,
        target_y,
        target_mean_f,
        target_std_f,
    ):
        optimality_vals = ADMMBO.get_optimality_objective(
            target_y, target_xz, zts, yts, rho
        )
        best_optimality_val = torch.max(optimality_vals)
        estimator = target_xz[torch.argmax(optimality_vals)]

        optimality_mean_f = ADMMBO.get_optimality_objective(
            target_mean_f, self.xz_domain, zts, yts, rho
        )

        ei_vals = expected_improvement(
            best_optimality_val, optimality_mean_f, target_std_f
        )
        query_idx = torch.argmax(ei_vals)
        return query_idx, estimator.reshape(1, self.dim)

    def get_constraint_query(
        self,
        rho,
        xt_plus_1,
        zti,
        yti,
        constraint_i_model,
        constraint_i_mean_f,
        constraint_i_std_f,
        inequality_i_type,
        constraint_i_xz,
        threshold_i,
    ):
        f_preds = GP.predict_f(constraint_i_model, constraint_i_xz)
        f_means = f_preds.mean
        f_stds = torch.sqrt(f_preds.variance)
        if inequality_i_type == "less_than_equal_to":
            f_means = -f_means

        past_feasibilities = gaussian_cdf(
            (threshold_i - f_means) / f_stds
        ) + rho / 2.0 / self.M * torch.square(
            torch.norm(xt_plus_1 - constraint_i_xz + yti, dim=1)
        )

        min_feasibility = torch.min(past_feasibilities)
        min_feasbility_idx = torch.argmin(past_feasibilities)
        estimator = constraint_i_xz[min_feasbility_idx]

        qti = (
            rho
            / 2.0
            / self.M
            * torch.square(torch.norm(xt_plus_1 - self.xz_domain + yti / rho, dim=1))
        )
        unsatisfaction_prob = gaussian_cdf(
            (threshold_i - constraint_i_mean_f) / constraint_i_std_f
        )
        satisfaction_prob = 1.0 - unsatisfaction_prob

        ei_vals = (
            torch.max(torch.tensor(0.0), min_feasibility - qti - 1.0)
            * unsatisfaction_prob
            + torch.max(torch.tensor(0.0), min_feasibility - qti) * satisfaction_prob
        )

        query_idx = torch.argmax(ei_vals)
        return query_idx, estimator

    def get_estimator(
        self,
        xz_domain,
        target_model,
        constraint_info_list,
        constraint_model_dict,
        cidx_to_moidx,
        threshold_list,
    ):
        f_preds = GP.predict_f(target_model, xz_domain)
        target_mean_f = f_preds.mean

        feasible_cond = torch.ones_like(target_mean_f, dtype=torch.bool)

        for i, threshold in enumerate(threshold_list):
            constraint_model = constraint_model_dict[cidx_to_moidx[i]]
            f_preds = GP.predict_f(constraint_model, xz_domain)
            constraint_mean_f = f_preds.mean
            constraint_std_f = torch.sqrt(f_preds.variance)

            if constraint_info_list[i]["inequality"] == "less_than_equal_to":
                constraint_mean_f = -constraint_mean_f

            constraint_satisfaction_prob = 1.0 - gaussian_cdf(
                (threshold - constraint_mean_f) / constraint_std_f
            )

            feasible_cond = torch.logical_and(
                feasible_cond, constraint_satisfaction_prob >= 1.0 - self.delta
            )

        feasible_idxs = feasible_cond.nonzero()
        if len(feasible_idxs) > 0:
            return xz_domain[
                feasible_idxs[torch.argmax(target_mean_f[feasible_idxs])]
            ].reshape(1, -1)
        else:
            return xz_domain[torch.argmax(target_mean_f)]

        return None
