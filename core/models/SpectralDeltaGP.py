import gpytorch
import torch

class SpectralDeltaGP(gpytorch.models.ExactGP):
    def __init__(self, input_dims, num_deltas, inducing_points_init=None):
        """
        :param input_dims: 输入维度大小
        :param num_deltas: delta 数量
        :param inducing_points_init: 诱导点的初始化，可以选填
        """
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-11))
        likelihood.register_prior("noise_prior", gpytorch.priors.HorseshoePrior(0.1), "noise")
        likelihood.noise = 1e-2

        # 在调用父类构造函数时，不传入训练数据，在 forward 时传入
        super(SpectralDeltaGP, self).__init__(train_inputs=None, train_targets=None, likelihood=likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        base_covar_module = gpytorch.kernels.SpectralDeltaKernel(
            num_dims=input_dims,  # 根据输入维度设置 Spectral Delta kernel
            num_deltas=num_deltas,
        )

        if inducing_points_init is not None:
            base_covar_module.initialize_from_data(inducing_points_init)

        self.covar_module = gpytorch.kernels.ScaleKernel(base_covar_module)

    def forward(self, x):
        # 根据输入的 x 进行预测
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
