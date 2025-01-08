import os
import math
import torch


import gpytorch

from matplotlib import pyplot as plt


class GP(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        initialization=None,
        prior=None,
        ard=True,
        fix_mean_at=None,
    ):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-3),
        )

        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        input_dim = train_x.shape[1]
        assert input_dim > 0

        ard_num_dims = input_dim if ard else None

        if prior is None:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
            )
        else:
            self.likelihood.noise_covar.register_prior(
                "noise_std_prior",
                prior["noise_std"],
                lambda module: module.noise.sqrt(),
            )

            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=ard_num_dims, lengthscale_prior=prior["lengthscale"]
                ),
                outputscale_prior=prior["outputscale"],
            )
            self.covar_module.base_kernel.register_constraint(
                "raw_lengthscale", gpytorch.constraints.GreaterThan(5e-2)
            )

        if initialization is None:
            if ard:
                if prior:
                    if (
                        len(prior["lengthscale"].mean.shape) == 0
                        or prior["lengthscale"].mean.squeeze().shape[0] == 1
                    ):
                        init_lengthscale = torch.squeeze(
                            prior["lengthscale"].mean
                        ) * torch.ones(1, input_dim)
                    else:
                        init_lengthscale = prior["lengthscale"].mean
                else:
                    init_lengthscale = torch.ones(1, ard_num_dims)

                self.covar_module.base_kernel.lengthscale = init_lengthscale
            else:
                self.covar_module.base_kernel.lengthscale = (
                    prior["lengthscale"].mean if prior else 1.0
                )

            self.covar_module.outputscale = (
                prior["outputscale"].mean if prior is not None else 1.0
            )
            self.likelihood.noise_covar.noise = 0.001

            self.mean_module.constant = 0.0

            if fix_mean_at is not None:
                self.mean_module.constant = fix_mean_at
                self.mean_module.constant.requires_grad = False

        else:
            self.initialize(**initialization)

        print("All constraints:")
        for constraint_name, constraint in self.named_constraints():
            print(f"Constraint name: {constraint_name:55} constraint = {constraint}")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @staticmethod
    def get_default_hyperparameter_prior(function_name="gausscurve"):
        return {
            "lengthscale": gpytorch.priors.GammaPrior(0.25, 0.5),
            "outputscale": gpytorch.priors.GammaPrior(2.0, 0.15),
            "noise_std": gpytorch.priors.NormalPrior(0.0, 0.1),
        }

    def save(self, path="model_state.pth"):
        torch.save(self.state_dict(), path)

    def plot1d(self, ax, x):
        assert x.shape[1] == 1

        f_preds = GP.predict_f(self, x)

        with torch.no_grad():
            f_means = f_preds.mean
            f_vars = f_preds.variance
            f_stds = torch.sqrt(f_vars)

        ax.fill_between(
            x.squeeze(),
            f_means - f_stds,
            f_means + f_stds,
            alpha=0.5,
        )
        ax.plot(x.squeeze(), f_means)

        ax.scatter(self.train_inputs[0].squeeze(), self.train_targets.squeeze())
        return ax

    @staticmethod
    def load(model, path="model_state.pth"):
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)

    @staticmethod
    def optimize_hyperparameters(
        model, train_x, train_y, learning_rate=0.1, training_iter=50, verbose=True
    ):
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            if verbose:
                print(
                    f"Iter {i+1}/{training_iter} - Loss: {loss.item():.3f} lengthscale: {model.covar_module.base_kernel.lengthscale}  noise: {model.likelihood.noise}"
                )
            optimizer.step()

    @staticmethod
    def predict_f(model, test_x):
        with torch.no_grad():
            model.eval()
            return model(test_x)

    @staticmethod
    def predict_y(model, test_x):
        with torch.no_grad():
            model.eval()
            return model.likelihood(model(test_x))
