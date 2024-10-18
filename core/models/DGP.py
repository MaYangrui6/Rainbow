import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from sklearn.cluster import KMeans
from torch import nn

from core.models.utils import initialize_inducing_points


class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='linear', inducing_points_init=None,
                 batch_shape=torch.Size([])):
        """
        诱导点的初始化现在可以根据输入数据的分布来确定
        :param inducing_points_init: 先验的诱导点初始化 (可选)，如果为 None，则根据输入数据分布初始化
        """
        if inducing_points_init is not None:
            # 如果提供了诱导点初始化的先验值，使用该值
            inducing_points = inducing_points_init
        else:
            # 如果没有提供诱导点初始化，使用默认方式根据数据初始化
            if output_dims is None:
                # 最后一层，单一输出，没有 batch_shape，随机生成诱导点
                inducing_points = torch.randn(num_inducing, input_dims)
            else:
                # 多任务或批处理的情况下，输出维度影响 batch_shape
                inducing_points = torch.randn(output_dims, num_inducing, input_dims)
                batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        # 变分推断是一种近似推断的方法，通过引入诱导点来构建变分分布，从而简化高斯过程的计算。具体地，变分推断将原始的高维协方差矩阵用低维的变分分布来近似，以降低计算复杂度。
        # 我们这里将 learn_inducing_locations 设置为 True, 也就是最大化变分下界（ELBO）来优化诱导点的位置
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        # 设置均值函数和协方差模块
        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)

        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        # 处理输入的均值和协方差矩阵，确保支持批处理
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # mean_x = torch.tanh(mean_x)
        # print('mean_x',mean_x)
        return MultivariateNormal(mean_x, covar_x)

    # def __call__(self, x, *other_inputs, **kwargs):
    #     """
    #     Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
    #     easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
    #     hidden layer's outputs and the input data to hidden_layer2.
    #     """
    #     if len(other_inputs):
    #         if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
    #             x = x.rsample()
    #
    #         # 处理其他输入的批处理扩展
    #         processed_inputs = [
    #             inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
    #             for inp in other_inputs
    #         ]
    #
    #         x = torch.cat([x] + processed_inputs, dim=-1)
    #
    #     return super().__call__(x, are_samples=bool(len(other_inputs)))



class DeepGPModel(DeepGP):
    def __init__(self, input_dims, num_layers=3, hidden_dims=None, num_tasks=1, inducing_points_init=None):
        """
        构建更深的深度高斯过程模型，允许多个隐藏层
        :param num_layers: 隐藏层的数量
        :param hidden_dims: 每层隐藏层的输出维度，可以是列表
        :param num_tasks: 多任务学习的任务数量
        :param inducing_points_init: 可选的诱导点初始化
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32] * (num_layers - 1)  # 如果未指定，默认所有隐藏层的输出维度相同

        self.hidden_layers = nn.ModuleList()

        # 第一个隐藏层
        batch_shape = torch.Size([num_tasks]) if num_tasks > 1 else torch.Size([])
        first_hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=input_dims,
            output_dims=hidden_dims[0],
            mean_type='linear',
            inducing_points_init=inducing_points_init,
            batch_shape=batch_shape
        )
        self.hidden_layers.append(first_hidden_layer)

        # 中间的隐藏层
        for i in range(1, num_layers - 1):
            hidden_layer = ToyDeepGPHiddenLayer(
                input_dims=hidden_dims[i - 1],
                output_dims=hidden_dims[i],
                mean_type='linear',
                batch_shape=batch_shape
            )
            self.hidden_layers.append(hidden_layer)

        # 最后一层
        last_layer = ToyDeepGPHiddenLayer(
            input_dims=hidden_dims[-1],
            output_dims=None,  # 最后一层不需要 output_dims
            mean_type='linear',
            batch_shape=batch_shape
        )
        self.last_layer = last_layer

        self.likelihood = GaussianLikelihood(batch_shape=batch_shape)

    def forward(self, inputs, are_samples=False):
        # 通过每一层进行前向传播，允许跳跃连接
        x = inputs
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x, are_samples=are_samples)

        # 最后一层
        output = self.last_layer(x, are_samples=are_samples)
        return output

        # 从 MultivariateNormal 提取均值
        # mean = output.mean
        # covariance_matrix = output.covariance_matrix

        # # 对均值应用 tanh 函数
        # mean_transformed = torch.sigmoid(mean)

        # # 创建新的 MultivariateNormal 对象，保留原始协方差
        # output_transformed = MultivariateNormal(mean_transformed, covariance_matrix)

        # return output_transformed


    def initialize_inducing_points_from_data(self, x_train, method='random', k=128):
        """
        初始化所有隐藏层的诱导点
        """
        for layer in self.hidden_layers:
            layer.initialize_inducing_points_from_data(x_train, method=method, k=k)

    def predict(self, test_loader, num_samples=10):
        """
        预测函数，处理批处理的输入，支持多样本采样
        :param test_loader: 测试数据加载器
        :param num_samples: 需要采样的数量，用于多次采样估计均值和不确定性
        :return: 多次采样的均值、方差和对数似然
        """
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                batch_mus = []
                batch_variances = []

                # 对于每个输入，采样 num_samples 次
                for _ in range(num_samples):
                    preds = self.likelihood(self(x_batch, are_samples=True))
                    batch_mus.append(preds.mean)
                    batch_variances.append(preds.variance)

                # 平均多个样本的预测结果
                mus.append(torch.stack(batch_mus, dim=0).mean(dim=0))
                variances.append(torch.stack(batch_variances, dim=0).mean(dim=0))
                lls.append(self.likelihood.log_marginal(y_batch, self(x_batch)))

        # 返回拼接的预测结果
        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)
