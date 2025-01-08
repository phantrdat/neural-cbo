import torch
import sys


class UCBC:
    def __init__(self):
        self.lowest_bound = float("inf")
        self.past_query_idxs = []
        self.estimator = None
        self.estimator_idx = None

    @staticmethod
    def get_Sminus(
        constraint_lower_f_list,
        constraint_upper_f_list,
        lower_margin_list,
        upper_margin_list,
    ):
        Sminus_cond = torch.ones_like(constraint_lower_f_list[0], dtype=torch.bool)

        for i in range(len(constraint_lower_f_list)):
            constraint_lower_f = constraint_lower_f_list[i]
            constraint_upper_f = constraint_upper_f_list[i]
            lower_margin = lower_margin_list[i]
            upper_margin = upper_margin_list[i]

            Sminus_cond_i = torch.logical_and(
                constraint_lower_f >= lower_margin, constraint_upper_f >= upper_margin
            )
            Sminus_cond = torch.logical_and(Sminus_cond, Sminus_cond_i)

        Sminus_idxs = Sminus_cond.nonzero()
        return Sminus_idxs, Sminus_cond

    @staticmethod
    def get_Sunion(constraint_upper_f_list, upper_margin_list):
        Sunion_cond = torch.ones_like(constraint_upper_f_list[0], dtype=torch.bool)

        for i in range(len(constraint_upper_f_list)):
            constraint_upper_f = constraint_upper_f_list[i]
            upper_margin = upper_margin_list[i]

            current_constraint_cond = constraint_upper_f >= upper_margin
            Sunion_cond = torch.logical_and(Sunion_cond, current_constraint_cond)

            if len(current_constraint_cond.nonzero()) == 0:
                print(f"get_Sunion: Constraint {i} is violated!")

        Sunion_idxs = Sunion_cond.nonzero()
        return Sunion_idxs, Sunion_cond

    def get_query(self, threshold_list, target, constraint, other_info):
        target_mean_f = target["mean_f"]
        target_std_f = target["std_f"]
        target_upper_f = target["upper_f"]
        target_lower_f = target["lower_f"]

        constraint_mean_f_list = constraint["mean_f_list"]
        constraint_std_f_list = constraint["std_f_list"]
        constraint_upper_f_list = constraint["upper_f_list"]
        constraint_lower_f_list = constraint["lower_f_list"]

        lower_margin_list = None
        upper_margin_list = None

        upper_margin_list = threshold_list

        Sunion_idxs, Sunion_cond = UCBC.get_Sunion(
            constraint_upper_f_list, upper_margin_list
        )

        current_bound = float("inf")

        if len(Sunion_idxs) == 0:
            max_violation = torch.tensor(0.0)
            query_idx = None
            query_type = []

            violations = []
            for i in range(len(constraint_upper_f_list)):
                constraint_upper_f = constraint_upper_f_list[i]
                upper_margin = upper_margin_list[i]

                violations.append(upper_margin - constraint_upper_f)

            violations = torch.stack(violations)
            max_violations, max_violations_idxs = torch.max(violations, dim=0)
            query_idx = torch.argmin(max_violations)
            query_type = [max_violations_idxs[query_idx].numpy() + 1]

            if query_idx is None:
                raise Exception(
                    "Strange: S+ is empty while cannot do BO on the constraint!"
                )
            method = "BO - CONSTRAINT"

        else:
            xplus_splus_idx = Sunion_idxs[torch.argmax(target_upper_f[Sunion_idxs])]
            query_idx = xplus_splus_idx

            method = "v2-coupled"
            query_type = [0] + list(range(1, len(threshold_list) + 1))

            estimated_regret = torch.max(target_mean_f + target_std_f) - (
                target_mean_f - target_std_f
            )
            for i, threshold in enumerate(threshold_list):
                constraint_upper_f = constraint_upper_f_list[i]
                constraint_lower_f = constraint_lower_f_list[i]

                violation = torch.nn.ReLU()(
                    torch.squeeze(
                        threshold
                        - (constraint_mean_f_list[i] - constraint_std_f_list[i])
                    )
                )
                estimated_regret = estimated_regret + violation

            self.estimator_idx = Sunion_idxs[
                torch.argmin(estimated_regret[Sunion_idxs])
            ]

        self.past_query_idxs.append(query_idx)

        return (
            query_idx,
            query_type,
            method,
            {
                "lower_margin_list": None,
                "upper_margin_list": None,
            },
        )
