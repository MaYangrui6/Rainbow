import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution

from core.models.utils import initialize_inducing_points


class DKLGPModel(ApproximateGP):
    """使用深度核学习 (DKL) 的高斯过程模型."""

    def __init__(self, input_dims, num_inducing=128, batch_shape=torch.Size([])):
        inducing_points = self.initialize_inducing_points_from_data(
            torch.randn(num_inducing, input_dims))

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        # 创建一个变分策略来管理诱导点
        variational_strategy = VariationalStrategy(
            self, inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=True
        )
        super(DKLGPModel, self).__init__(variational_strategy)

        # 核函数基于特征提取器输出的 RBF 核
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        self.likelihood = GaussianLikelihood(batch_shape=batch_shape)

    def forward(self, x):
        # 从图编码器中提取特征
        projected_x = self.scale_to_bounds(x)  # Make the NN values "nice"

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return MultivariateNormal(mean_x, covar_x)

    def initialize_inducing_points_from_data(self, x_train, method='kmeans', k=128):
        """
        :param x_train: 训练数据
        :param method: 初始化诱导点的方法，可以是 'random' 或 'kmeans'
        :param k: 诱导点数量
        """
        return initialize_inducing_points(x_train, method=method, k=k)