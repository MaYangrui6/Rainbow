import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, TopKPooling, MessagePassing
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.loader import DataLoader
import gpytorch
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood
from torch_geometric.utils import add_self_loops, degree

from core.models.DGP import DeepGPModel
from core.models.DKL import DKLGPModel
from sql import Sql, _global
from core import database, Sql, Plan, load
import torch.nn.functional as F


def dgl_to_pyg(dgl_graph):
    """将 DGL 图转换为 PyG 格式."""
    x = dgl_graph.ndata['feature']  # 节点特征
    edge_index = torch.stack(dgl_graph.edges()).long()  # 边索引 (转换为 PyG 格式)
    edge_attr = dgl_graph.edata['feature']  # 边特征
    batch = torch.zeros(dgl_graph.number_of_nodes(), dtype=torch.long)  # 所有节点归为同一批次
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


class CustomTransformerConv(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5, heads=2, edge_dim=None):
        super(CustomTransformerConv, self).__init__(aggr='add')  # 'add' 聚合方式
        self.linear = nn.Linear(in_channels, out_channels)
        self.transformer_conv = TransformerConv(in_channels, out_channels, heads=heads, edge_dim=edge_dim, dropout_rate=dropout_rate)

    def forward(self, x, edge_index, edge_attr, edge_weight):
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        x = self.linear(x)

        # Step 1: 处理边权重。使用 edge_weight 作为消息传递中的归一化因子。该部分主要控制节点间信息传递的强度，通过节点间的边权重来加权调整节点特征的聚合。
        x_weighted = self.propagate(edge_index, x=x, norm=norm)

        # Step 2: 处理边特征。 使用 TransformerConv 处理边特征并结合多头注意力机制
        x_transformed = self.transformer_conv(x, edge_index, edge_attr=edge_attr)

        # 合并两步处理的结果，也可以尝试其他结合方式，比如拼接或加权融合
        out = x_weighted + x_transformed
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GraphEncoder(nn.Module):
    """图编码模块，基于 CustomTransformerConv."""

    def __init__(self, in_feats, edge_feats, embedding_size, num_layers=3, top_k_every_n=3, top_k_ratio=0.5,
                 n_heads=4, dropout_rate=0.5):
        super(GraphEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.top_k_every_n = top_k_every_n
        self.edge_dim = edge_feats
        self.n_heads = n_heads

        # 第一层 CustomTransformerConv
        self.conv1 = CustomTransformerConv(in_feats, self.embedding_size, heads=n_heads, edge_dim=edge_feats, dropout_rate=dropout_rate)
        self.transf1 = nn.Linear(self.embedding_size * n_heads, self.embedding_size)
        self.bn1 = nn.BatchNorm1d(self.embedding_size)

        # 其他 CustomTransformerConv 层及 TopKPooling
        self.conv_layers = nn.ModuleList()
        self.transf_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.conv_layers.append(
                CustomTransformerConv(self.embedding_size, self.embedding_size, heads=n_heads, edge_dim=edge_feats))
            self.transf_layers.append(nn.Linear(self.embedding_size * n_heads, self.embedding_size))
            self.bn_layers.append(nn.BatchNorm1d(self.embedding_size))
            self.pooling_layers.append(TopKPooling(self.embedding_size, ratio=top_k_ratio))

    def forward(self, x, edge_attr, edge_index, edge_weight, batch_index):
        # 这里的 edge_weight 用标准化后的 Promotion 来指定。
        x = self.conv1(x, edge_index, edge_attr, edge_weight)

        # 调整线性层输入的形状
        x = torch.relu(self.transf1(x.view(-1, self.embedding_size * self.n_heads)))
        x = self.bn1(x)

        global_representation = []

        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, edge_attr, edge_weight)
            x = torch.relu(self.transf_layers[i](x.view(-1, self.embedding_size * self.n_heads)))  # 确保形状匹配
            x = self.bn_layers[i](x)

            if i % self.top_k_every_n == 0 or i == len(self.conv_layers) - 1:
                x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[i](
                    x, edge_index, edge_attr, batch_index
                )
                global_representation.append(torch.cat([
                    gmp(x, batch_index),
                    gap(x, batch_index)
                ], dim=1))

        x = sum(global_representation)

        return x


class ImprovementPredictionModel(nn.Module):
    """预测模型，基于图编码."""
    def __init__(self, in_feats, edge_feats, graph_embedding_size, use_dgp=True):
        super(ImprovementPredictionModel, self).__init__()
        self.use_dgp = use_dgp
        self.graph_encoder = GraphEncoder(in_feats, edge_feats, graph_embedding_size)

        # 定义高斯过程模型和对应的似然函数
        self.likelihood = GaussianLikelihood()
        if self.use_dgp:
            self.gp_model = DeepGPModel(
                graph_embedding_size * 2)  # 2 * graph_embedding_size 是因为我们在全局池化时合并了 mean 和 max 结果
        else:
            self.gp_model = DKLGPModel(graph_embedding_size * 2)

    def forward(self, x, edge_attr, edge_index, batch_index):
        graph_embedding = self.graph_encoder(x, edge_attr, edge_index, batch_index)

        # 确保在训练时调用 train()，在预测时调用 eval()
        self.gp_model.train()
        self.likelihood.train()

        with gpytorch.settings.fast_pred_var():
            # 使用 graph_embedding 进行预测
            output = self.gp_model(graph_embedding)
            prediction = self.likelihood(output)

        return prediction  # 返回高斯过程的均值作为预测值

    def predict(self, x, edge_attr, edge_index, batch_index):
        """在预测时调用"""
        graph_embedding = self.graph_encoder(x, edge_attr, edge_index, batch_index)
        self.gp_model.eval()
        self.likelihood.eval()

        with gpytorch.settings.fast_pred_var():
            output = self.gp_model(graph_embedding)
            prediction = self.likelihood(output)

        return prediction


def load_sql_graphs_pyg(csv_path, dbname, user, password, host, port, start_idx=5000, end_idx=5050):
    """
    从指定的 CSV 文件加载 SQL 查询，并将它们转换为 PyG 图数据。

    参数:
    - csv_path (str): CSV 文件路径。
    - dbname (str): 数据库名称。
    - user (str): 数据库用户。
    - password (str): 数据库密码。
    - host (str): 数据库主机地址。
    - port (str): 数据库端口。
    - start_idx (int): SQL 查询的起始索引。
    - end_idx (int): SQL 查询的结束索引。

    返回:
    - sql_graphs_pyg (list): PyG 数据列表。
    """
    # 读取 CSV 数据
    df = pd.read_csv(csv_path).iloc[:, 3:]

    # 设置数据库连接
    database.setup(dbname=dbname, user=user, password=password, host=host, port=port, cache=False)

    sql_graphs_pyg = []

    # 遍历指定范围的 SQL 查询
    for sql_query in df['query'].values[start_idx:end_idx]:
        sql_instance = Sql(sql_query)
        g, data_dict, node_indexes = sql_instance.to_hetero_graph_dgl()

        # 将节点特征拼接
        combined_features = torch.cat(
            [g.ndata[key].view(g.ndata['onehot'].shape[0], -1) if key == 'others' else g.ndata[key] for key in
             ['filter', 'edge', 'onehot', 'others']], dim=1)

        g.ndata['feature'] = combined_features

        # 随机生成边特征
        g.edata['feature'] = torch.randn(g.num_edges(), num_edge_features)

        # 转换为 PyG 数据
        pyg_data = dgl_to_pyg(g)
        sql_graphs_pyg.append(pyg_data)

    return sql_graphs_pyg


if __name__ == "__main__":
    num_edge_features = 4

    # 读取数据并设置数据库
    sql_graphs_pyg = load_sql_graphs_pyg(
        csv_path='/home/ubuntu/project/mayang/Classification/process_data/tpcds/dataset_tpcds_383_plan.csv',
        dbname='indexselection_tpcds___1',
        user='postgres',
        password='password',
        host='127.0.0.1',
        port='5432',
        start_idx=5000,
        end_idx=5050
    )

    # 创建 PyG DataLoader
    data_loader = DataLoader(sql_graphs_pyg, batch_size=32, shuffle=True)

    model = ImprovementPredictionModel(sql_graphs_pyg[0].x.shape[1], sql_graphs_pyg[0].edge_attr.shape[1],
                                       graph_embedding_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    mll = DeepApproximateMLL(VariationalELBO(model.gp_model.likelihood, model.gp_model, data_loader.batch_size))

    # # 定义Explainer
    # # 定义 model_config，指定输入模式和解释的目标（节点或边）
    # model_config = ModelConfig(
    #     mode='regression',  # 任务类型，回归或分类
    #     task_level='graph',  # 解释任务级别，可以是 'node', 'edge', or 'graph'
    #     return_type='raw',  # 模型的输出格式，'log_probs', 'probs' 或 'raw'。'raw' 用于直接的输出预测
    # )
    #
    # explainer = Explainer(
    #     model=model.graph_encoder,  # 解释模型的图编码部分
    #     algorithm=GNNExplainer(epochs=200),  # 使用 GNNExplainer 算法
    #     model_config=model_config,
    #     explanation_type='model',  # 解释模型的现象
    #     node_mask_type='object',  # 节点特征掩码
    #     edge_mask_type='object',  # 边特征掩码
    #     # threshold_config={'hard_threshold': 0.5},  # 设置阈值来筛选重要的边或节点
    # )

    # 训练循环
    for epoch in range(500):
        model.train()
        total_loss = 0
        for batch in data_loader:
            # 前向传播
            with gpytorch.settings.num_likelihood_samples(data_loader.batch_size):
                pred_time = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)

                # 假设目标是生成随机数据，你需要用实际的目标数据替换
                target = torch.randn(data_loader.batch_size, 1, dtype=torch.float)

                # 计算损失
                # loss = criterion(pred_time, target)
                loss = -mll(pred_time, target)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {total_loss:.4f}')

        # # 每10轮进行一次预测和评估
        # if (epoch + 1) % 10 == 0:
        #     model.eval()
        #     test_lls = []
        #     all_pred = []
        #     all_target = []
        #
        #     with torch.no_grad():
        #         for batch in data_loader:
        #             pred_time = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)
        #
        #             # 获取均值（预测值）
        #             pred_mean = pred_time.mean
        #
        #             # 计算负对数似然（NLL）
        #             test_ll = -mll(pred_time, batch.y)  # 假设真实目标值为 batch.y
        #             test_lls.append(test_ll.item())
        #
        #             # 收集预测值和目标值，用于计算 RMSE
        #             all_pred.append(pred_mean.cpu())
        #             all_target.append(batch.y.cpu())
        #
        #     # 将所有批次的预测值和目标值拼接起来
        #     all_pred = torch.cat(all_pred, dim=0)
        #     all_target = torch.cat(all_target, dim=0)
        #
        #     # 计算 RMSE
        #     rmse = torch.sqrt(F.mse_loss(all_pred, all_target))
        #
        #     # 输出 RMSE 和 NLL
        #     print(f"RMSE: {rmse.item()}, NLL: {-torch.tensor(test_lls).mean().item()}")

    # # 解释模型
    # batch = next(iter(data_loader))
    # explanation = explainer(x=batch.x, edge_attr=batch.edge_attr, edge_index=batch.edge_index, index=0)
    #
    # # 可视化解释结果
    # G = to_networkx(batch, node_attrs=['x'], edge_attrs=['edge_attr'])
    # node_mask = explanation.node_mask
    # edge_mask = explanation.edge_mask
    #
    # plt.figure(figsize=(8, 8))
    # nx.draw(G, node_color=node_mask, edge_color=edge_mask, with_labels=True, cmap='coolwarm')
    # plt.show()
