import pickle

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.loader import DataLoader
from blitz.utils import variational_estimator
from blitz.modules import BayesianLinear
from torch.optim.lr_scheduler import StepLR
import os
import sys
import random

loger_dir = os.path.abspath('/home/ubuntu/project/mayang/LOGER')

from core.models.DGP import DeepGPModel
from core.models.DKL import DKLGPModel
from sql import Sql, _global
from core import database, Sql, Plan, load
import ast
import re
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv, TopKPooling
from torch_geometric.nn import GATConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gmp, global_add_pool as gap

# 获取 QPE 目录的绝对路径
qpe_dir = os.path.abspath('/home/ubuntu/project/rainbow')

# 将 QPE 目录添加到 sys.path
sys.path.append(qpe_dir)

# 现在尝试导入 TreeBuilder 类
from QPE.sql2fea import TreeBuilder


def clean_expression(expr):
    """清理表达式中的多余空格和括号"""
    # 移除首尾的空格
    expr = expr.strip()

    # 移除最外层的括号，如果存在
    if expr.startswith('('):
        expr = expr[1:].strip()
    if expr.endswith(')'):
        expr = expr[:-1].strip()

    return expr


def split_cmp_expression(cmp, num):
    '''将含有 OR 或 AND 的表达式分割并提取每个子表达式，并保持原来的编号'''

    # 移除最外层的括号
    cmp = cmp.strip()
    if cmp.startswith('(') and cmp.endswith(')'):
        cmp = cmp[1:-1].strip()

    # 按 OR 和 AND 分割
    parts = re.split(r'(\s+OR\s+|\s+AND\s+)', cmp)

    results = []
    current_num = num

    for part in parts:
        part = clean_expression(part)  # 清理表达式

        # 忽略分隔符 'OR' 和 'AND'
        if part in ['OR', 'AND']:
            continue

        # 递归处理有内部括号的部分
        if part.startswith('(') and part.endswith(')'):
            nested_result = split_cmp_expression(part, current_num)
            results.extend(nested_result)
        else:
            results.append((part, current_num))

    return results


def parse_attribute_config(index_str):
    '''处理输入的configuration，转化成cols/attribute的形式'''
    # 将字符串转换为列表
    index_list = ast.literal_eval(index_str)

    table_cols_list = []
    for index in index_list:
        # 去掉 I() 的部分，提取表名和列名
        index_content = index[4:-1]  # 去掉前面的 'I(' 和后面的 ')'
        table_cols = index_content.split(',')
        table_cols_list.append(table_cols)

    return table_cols_list


def swap_join_condition(condition: str) -> str:
    '''防止同一个连接条件，左右连接条件不一样而无法识别'''
    # 找到等号的位置
    equal_index = condition.find('=')

    # 如果没有找到等号，返回原字符串
    if equal_index == -1:
        return condition

    # 拆分等号两边的条件，并去掉左右空格
    left_side = condition[:equal_index].strip()
    right_side = condition[equal_index + 1:].strip()

    # 交换等式两边的条件
    swapped_condition = f"{right_side} = {left_side}"

    return swapped_condition


def has_text_expression(cmp: str) -> bool:
    '''判断cmp中有没有文本'''
    # 使用正则表达式查找被引号包围的文本
    pattern = r"'[^']*'"

    # 查找匹配的内容
    match = re.search(pattern, cmp)

    # 如果找到了匹配内容，返回 True；否则返回 False
    return match is not None


def convert_between_expression(cmp: str) -> str:
    '''转换 BETWEEN 表达式'''
    match = re.match(r"([\w\.]+)\.(\w+)\s*BETWEEN\s*(.*)\s*AND\s*(.*)", cmp)

    if not match:
        print('The input string is not a valid BETWEEN expression. cmp', cmp)
        return cmp.split('.')
        # raise ValueError("The input string is not a valid BETWEEN expression.")

    table_name, column, lower_bound, upper_bound = match.groups()

    # 检查值是否是数字
    if lower_bound.strip().isdigit() and upper_bound.strip().isdigit():
        new_expression = f"(({column} >= {lower_bound.strip()}) AND ({column} <= {upper_bound.strip()}))"
    else:
        new_expression = f"(({column})::text >= {lower_bound.strip()}::text) AND (({column})::text <= {upper_bound.strip()}::text)"

    return table_name, new_expression[:-1]


def convert_to_text_comparison(cmp: str):
    '''转化cmp与plan中的谓词表示，处理 IN 和复杂的 OR 语句。'''

    # 去除外部括号和空白
    cmp = cmp.strip()
    if cmp.startswith('(') and cmp.endswith(')'):
        cmp = cmp[1:-1].strip()

    # 处理 OR 语句，切割并处理第一个部分
    if ' OR ' in cmp:
        first_or_index = cmp.index(' OR ')
        first_part = cmp[:first_or_index].strip()
        return convert_to_text_comparison(first_part)

    # 处理 AND 语句，切割并处理第一个部分
    if ' AND ' in cmp:
        first_and_index = cmp.index(' AND ')
        first_part = cmp[:first_and_index].strip()
        return convert_to_text_comparison(first_part)

    # 处理 '=' 和 IN 语句
    match_eq = re.match(r"\(?\s*([\w\.]+)\.(\w+)\s*(=|<>|<|>|<=|>=)\s*'([^']*)'\s*\)?", cmp)
    match_in = re.match(r"\(?\s*([\w\.]+)\.(\w+)\s*IN\s*\((.*?)\)\s*\)?", cmp, re.DOTALL)

    if match_eq:
        # 处理 '=' 等值比较的情况
        table_name, column, operator, value = match_eq.groups()
        new_expression = f"(({column})::text {operator} '{value}'::text)"
        return table_name, new_expression

    elif match_in:
        # 处理 'IN' 语句，提取表名、列名以及 IN 中的值列表
        table_name, column, values = match_in.groups()

        # 使用正则提取 IN 中的值，允许值跨多行
        value_list = re.findall(r"'(.*?)'", values)
        if not value_list:
            return cmp.split('.')
            # raise ValueError("在 IN 表达式中未找到有效值。")

        # 单个值的情况：转化为等值比较
        if len(value_list) == 1:
            new_expression = f"(({column})::text = '{value_list[0]}'::text)"
        else:
            # 多个值的情况：转化为 ANY 比较
            formatted_values = ','.join(f"'{v}'" for v in value_list)
            new_expression = f"(({column})::text = ANY ('{{{formatted_values}}}'::text[]))"

        return table_name, new_expression

    # 无法匹配时，抛出错误
    print(f"无法匹配的输入: {cmp}")  # 调试信息
    raise ValueError("输入字符串不是有效的比较表达式。")


def trim_parentheses_and_spaces(input_str: str) -> str:
    # 去除开头的空格
    trimmed_str = input_str.lstrip()
    # 如果开头是 '('，则去除它
    if trimmed_str.startswith('('):
        trimmed_str = trimmed_str[1:].lstrip()  # 去掉 '(' 以及后面的空格
    return trimmed_str


def transform_sql_like_condition(sql_condition):
    '''处理LIKE和NOT LIKE条件'''
    # 匹配 `AND` 和 `OR`
    sql_condition = trim_parentheses_and_spaces(sql_condition)
    conditions = re.split(r"(?i) AND | OR ", sql_condition)
    # print(conditions)

    # 转换每个 LIKE 和 NOT LIKE 条件为 Postgres 的格式
    transformed_clauses = []
    for condition in conditions:
        condition = condition.strip()
        table_name = condition.split('.')[0]
        col_name = condition.split()[0][len(table_name) + 1:]
        if 'not like' in condition.lower():
            transformed_clause = f"(({col_name})::text !~~ {condition.split('not like')[1].strip().rstrip(')')}::text)"
        elif 'like' in condition.lower():
            transformed_clause = f"(({col_name})::text ~~ {condition.split('like')[1].strip().rstrip(')')}::text)"
            # print("condition.split('like')[1].strip()",condition.split('like')[1].strip())
        else:
            continue

        transformed_clauses.append(transformed_clause)

    # 返回表名和第一个转换后的条件
    return table_name, transformed_clauses[0]


# def delete_left_table_cond(sql_condition):
#     # 使用正则表达式匹配表名和列名
#     match = re.match(r"[\w\.]+\.(\w+)\s*=\s*(.*)", sql_condition)

#     if not match:
#         raise ValueError("The input string is not a valid comparison expression.")

#     # 提取列名和右侧表达式
#     column = match.group(1)
#     right_side = match.group(2)

#     return f"{column} = {right_side.strip()}"
def delete_left_table_cond(sql_condition):
    # 扩展正则表达式，支持更多运算符并处理可能的括号
    match = re.match(r"[\w\.]+\.(\w+)\s*(=|<>|<|>|<=|>=)\s*(.*)", sql_condition)

    if not match:
        print('sql_condition', sql_condition)
        raise ValueError("The input string is not a valid comparison expression.")

    # 提取列名、运算符和右侧表达式
    column = match.group(1)
    operator = match.group(2)
    right_side = match.group(3)

    # 返回去掉表名前缀的表达式
    return f"{column} {operator} {right_side.strip()}"


def dgl_node_and_edge_vectorization(sql_query, config_index, plan, attribute_dict):
    global predicte_num,join_num  # 声明使用全局变量
    num_edge_features = 15
    sql_instance = Sql(sql_query)

    # 构建异构图
    g, data_dict, node_indexes, edge_lists = sql_instance.to_hetero_graph_dgl()
    # 处理edge_list中一个语句存在很多and、or语句的情况
    edge_list = []
    for cmp, num_table in edge_lists:
        if num_table==1:
            predicte_num+=1
        else:
            join_num+=1
        if 'BETWEEN' in cmp:
            edge_list += [(cmp, num_table)]
        elif 'OR' in cmp or 'AND' in cmp:
            cmp_split = split_cmp_expression(cmp, num_table)
            edge_list += cmp_split
        else:
            edge_list += [(cmp, num_table)]
    # 构建 node feature
    filter_features = g.ndata['filter']
    edge_features = g.ndata['edge']
    onehot_features = g.ndata['onehot']
    others_features = g.ndata['others'].view(g.ndata['filter'].shape[0], -1)

    # 假设两个特征的维度是相同的
    combined_features = torch.cat((filter_features, edge_features, onehot_features, others_features), dim=1)
    g.ndata['feature'] = combined_features

    # 构建 edge feature
    tree_builder = TreeBuilder()
    tree_builder.set_configruations(config_index)
    tree_builder.set_table_to_alias_dict(sql_query)
    # 准备您的 SQL 执行计划数据，这里是一个示例
    execution_plan = ast.literal_eval(plan)[0]['Plan']

    # 特征化执行计划
    features = tree_builder.plan_to_feature_tree(execution_plan, current_height=0)

    # 提取对应operator vector
    operator_vector_dict = tree_builder.get_operator_vector()
    # 建立cmp与plan_operator的映射关系
    tables = []
    operator_vectors_cmp1 = []
    operator_vectors_cmp2 = []
    get_keys = []
    for cmp, table_num in edge_list:
        flag = False
        # print('cmp:',cmp)
        if table_num == 1:
            if 'like' in cmp and has_text_expression(cmp):
                try:
                    table, cmp_in_op = transform_sql_like_condition(cmp)
                except:
                    print('error process like_cmp:', cmp)
            elif 'BETWEEN' in cmp:
                table, cmp_in_op = convert_between_expression(cmp)
            elif has_text_expression(cmp):
                table, cmp_in_op = convert_to_text_comparison(cmp)
            else:
                table, cmp_in_op = cmp.split('.')

        for key in operator_vector_dict:
            if table_num == 1:
                table_op, key_op = key.split('_47_')
                if cmp_in_op in key and table == table_op:
                    tables.append(table)
                    operator_vectors_cmp1.append(operator_vector_dict[key])
                    flag = True
                    break
            # print(key)
            elif cmp in key or swap_join_condition(cmp) in key or delete_left_table_cond(
                    cmp) in key or delete_left_table_cond(swap_join_condition(cmp)) in key:
                get_keys.append(cmp)
                operator_vectors_cmp2.append(operator_vector_dict[key])
                operator_vectors_cmp2.append(operator_vector_dict[key])
                flag = True
                break
        if not flag:
            # 存在plan中不存在的cmp
            print('error cmp', cmp)
            if table_num == 2:
                print('add edge vector:', table_num)
                operator_vectors_cmp2.append(torch.zeros((1, num_edge_features)))
                operator_vectors_cmp2.append(torch.zeros((1, num_edge_features)))
    # print('cmp',len(edge_list))
    # print(len(operator_vectors_cmp1),len(operator_vectors_cmp2))
    # 合并每一条边的向量
    # 这个sql没有join关系：
    if len(operator_vectors_cmp1):
        vector_cmp1 = torch.cat(operator_vectors_cmp1, dim=0)
        vector_cmp2 = torch.cat(operator_vectors_cmp2, dim=0)
        vector_edge = torch.cat((vector_cmp2, vector_cmp1), dim=0)
    else:
        vector_edge = torch.cat(operator_vectors_cmp2, dim=0)
    # 添加自环
    for t in tables:
        new_u = torch.tensor([node_indexes['~' + t]])  # 自环的起点
        new_v = torch.tensor([node_indexes['~' + t]])  # 自环的终点
        g.add_edges(new_u, new_v)
    g.edata['feature'] = vector_edge

    # 构建configuration特征
    configuration_vector = [0] * len(attribute_dict)
    for attrs in parse_attribute_config(config_index):
        for p in range(len(attrs)):
            # 获得在indexed_attribute中的位置
            attr = attrs[p]
            if attr[:2] == 'C ':
                attr = attr[2:]
            # 构建configuration特征(剔除与sql不相关的index，不在sql.txt中的index)
            # if attr.split('.')[1] not in sql_query:
            #     continue
            attr_p = attribute_dict[attr]
            configuration_vector[attr_p] += 1 / (p + 1)
    g.global_data = {"configuration_vector": torch.tensor(configuration_vector)}
    return g


def dgl_to_pyg(dgl_graph, label):
    """将 DGL 图转换为 PyG 格式."""
    x = dgl_graph.ndata['feature']  # 节点特征
    edge_index = torch.stack(dgl_graph.edges()).long()  # 边索引 (转换为 PyG 格式)
    edge_attr = dgl_graph.edata['feature']  # 边特征
    configuration_vector = dgl_graph.global_data['configuration_vector']  # configuration_vector
    batch = torch.zeros(dgl_graph.number_of_nodes(), dtype=torch.long)  # 所有节点归为同一批次
    label_tensor = torch.tensor(label, dtype=torch.float32)  # 根据需要使用 float32 或其他类型
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, configuration_vector=configuration_vector,
                y=label_tensor, batch=batch)


class GraphEncoder(nn.Module):
    """图编码模块，基于 TransformerConv,添加了残差连接。"""

    def __init__(self, in_feats, edge_feats, embedding_size, num_layers=3, top_k_every_n=3, top_k_ratio=0.5, n_heads=4,
                 dropout_rate=0.5):
        super(GraphEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.top_k_every_n = top_k_every_n
        self.edge_dim = edge_feats
        self.n_heads = n_heads

        # 第一层 TransformerConv
        self.conv1 = TransformerConv(in_feats, self.embedding_size, heads=n_heads, dropout=dropout_rate,
                                     edge_dim=edge_feats, beta=True)
        self.transf1 = nn.Linear(self.embedding_size * n_heads, self.embedding_size)
        self.bn1 = nn.BatchNorm1d(self.embedding_size)

        # 其他 TransformerConv 层及 TopKPooling
        self.conv_layers = nn.ModuleList()
        self.transf_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.conv_layers.append(
                TransformerConv(self.embedding_size, self.embedding_size, heads=n_heads, dropout=dropout_rate,
                                edge_dim=edge_feats, beta=True))
            self.transf_layers.append(nn.Linear(self.embedding_size * n_heads, self.embedding_size))
            self.bn_layers.append(nn.BatchNorm1d(self.embedding_size))
            self.pooling_layers.append(TopKPooling(self.embedding_size, ratio=top_k_ratio))

    def forward(self, x, edge_attr, edge_index, batch_index):
        # 残差连接的初始输入
        residual = x

        # 第一个 TransformerConv 层及线性变换
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x.view(-1, self.embedding_size * self.n_heads)))
        x = self.bn1(x)

        # 添加残差连接，将初始输入加入 x
        if residual.shape == x.shape:
            x = x + residual  # 进行残差连接

        global_representation = []

        for i, conv_layer in enumerate(self.conv_layers):
            # 每一层的残差输入
            residual = x

            # TransformerConv 和线性变换
            x = conv_layer(x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x.view(-1, self.embedding_size * self.n_heads)))  # 确保形状匹配
            x = self.bn_layers[i](x)

            # 残差连接
            if residual.shape == x.shape:
                x = x + residual  # 进行残差连接

            if i % self.top_k_every_n == 0 or i == len(self.conv_layers) - 1:
                x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[i](
                    x, edge_index, edge_attr, batch_index
                )
                global_representation.append(torch.cat([
                    gmp(x, batch_index),
                    gap(x, batch_index)
                ], dim=1))

        # 将全局表示相加作为最终输出
        x = sum(global_representation)

        return x


class GraphEncoder_GATConv(nn.Module):
    def __init__(self, in_feats, edge_feats, embedding_size, num_layers=3, top_k_every_n=3, top_k_ratio=0.5, n_heads=4,
                 dropout_rate=0.5):
        super(GraphEncoder_GATConv, self).__init__()
        self.embedding_size = embedding_size
        self.top_k_every_n = top_k_every_n
        self.edge_dim = edge_feats
        self.n_heads = n_heads

        # 使用 GATConv 替代 GCNConv
        self.conv1 = GATConv(in_feats, self.embedding_size, heads=n_heads, dropout=dropout_rate)
        self.transf1 = nn.Linear(self.embedding_size * n_heads, self.embedding_size)  # *n_heads because GAT has multiple heads
        self.bn1 = nn.BatchNorm1d(self.embedding_size)

        # 其他 GATConv 层及池化层
        self.conv_layers = nn.ModuleList()
        self.transf_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.conv_layers.append(
                GATConv(self.embedding_size, self.embedding_size, heads=n_heads, dropout=dropout_rate)  # GATConv with multiple heads
            )
            self.transf_layers.append(nn.Linear(self.embedding_size * n_heads, self.embedding_size))
            self.bn_layers.append(nn.BatchNorm1d(self.embedding_size))
            self.pooling_layers.append(TopKPooling(self.embedding_size, ratio=top_k_ratio))

    def forward(self, x, edge_attr, edge_index, batch_index):
        # 前向传播保持不变
        residual = x

        # 第一层 GATConv
        x = self.conv1(x, edge_index, edge_attr)  # GATConv
        x = torch.relu(self.transf1(x.view(-1, self.embedding_size * self.n_heads)))  # Reshape due to multiple heads
        x = self.bn1(x)

        # 残差连接
        if residual.shape == x.shape:
            x = x + residual  # 进行残差连接

        global_representation = []

        for i, conv_layer in enumerate(self.conv_layers):
            residual = x
            x = conv_layer(x, edge_index, edge_attr)  # GATConv
            x = torch.relu(self.transf_layers[i](x.view(-1, self.embedding_size * self.n_heads)))  # Reshape
            x = self.bn_layers[i](x)

            # 残差连接
            if residual.shape == x.shape:
                x = x + residual

            if i % self.top_k_every_n == 0 or i == len(self.conv_layers) - 1:
                x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[i](
                    x, edge_index, edge_attr, batch_index
                )
                global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))

        # 全局表示
        x = sum(global_representation)

        return x


class GraphEncoder_ChebConv(nn.Module):
    """图编码模块，基于 ChebConv，添加了残差连接。"""

    def __init__(self, in_feats, edge_feats, embedding_size, num_layers=3, top_k_every_n=3, top_k_ratio=0.5, n_heads=4,
                 dropout_rate=0.5, K=3):
        super(GraphEncoder_ChebConv, self).__init__()
        self.embedding_size = embedding_size
        self.top_k_every_n = top_k_every_n
        self.edge_dim = edge_feats
        self.n_heads = n_heads
        self.K = K  # 设置 ChebConv 的阶数

        # 第一层 ChebConv
        self.conv1 = ChebConv(in_feats, self.embedding_size, K=self.K)
        self.transf1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.bn1 = nn.BatchNorm1d(self.embedding_size)

        # 其他 ChebConv 层及 TopKPooling
        self.conv_layers = nn.ModuleList()
        self.transf_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.conv_layers.append(
                ChebConv(self.embedding_size, self.embedding_size, K=self.K)  # 使用 ChebConv 替代 TransformerConv
            )
            self.transf_layers.append(nn.Linear(self.embedding_size, self.embedding_size))
            self.bn_layers.append(nn.BatchNorm1d(self.embedding_size))
            self.pooling_layers.append(TopKPooling(self.embedding_size, ratio=top_k_ratio))

    def forward(self, x, edge_attr, edge_index, batch_index):
        # 前向传播保持不变
        residual = x

        # 第一层
        x = self.conv1(x, edge_index)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        # 残差连接
        if residual.shape == x.shape:
            x = x + residual  # 进行残差连接

        global_representation = []

        for i, conv_layer in enumerate(self.conv_layers):
            residual = x
            x = conv_layer(x, edge_index)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)

            # 残差连接
            if residual.shape == x.shape:
                x = x + residual

            if i % self.top_k_every_n == 0 or i == len(self.conv_layers) - 1:
                x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[i](
                    x, edge_index, edge_attr, batch_index
                )
                global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))

        # 全局表示
        x = sum(global_representation)

        return x



@variational_estimator
class ImprovementPredictionModelGNN(nn.Module):
    """使用图卷积神经网络进行预测，基于图编码，并增加多层贝叶斯线性层，融合配置向量."""

    def __init__(self, in_feats, edge_feats, graph_embedding_size, config_vector_size, hidden_dim=128, num_layers=3):
        print('start ImprovementPredictionModelGNN')
        super(ImprovementPredictionModelGNN, self).__init__()

        # 图编码器
        self.graph_encoder = GraphEncoder_GATConv(in_feats, edge_feats, graph_embedding_size)

        # 定义贝叶斯线性层
        input_dim = graph_embedding_size * 2 + config_vector_size
        self.blinear1 = BayesianLinear(input_dim, hidden_dim, prior_sigma_1=1)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)

        # 中间的隐藏层
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(BayesianLinear(hidden_dim, hidden_dim, prior_sigma_1=1))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # 最后一层，隐藏层到输出维度

        self.blinear_out = BayesianLinear(hidden_dim, 1, prior_sigma_1=1)  # 输出为单个值

    def forward(self, x, edge_attr, edge_index, configuration_vector, batch_index):
        # 通过图编码器获取图的嵌入表示
        graph_embedding = self.graph_encoder(x, edge_attr, edge_index, batch_index)

        # 拼接 graph_embedding 和 configuration_vector
        configuration_vector = configuration_vector.view(graph_embedding.size(0), -1)
        graph_embedding = torch.cat([graph_embedding, configuration_vector], dim=1)

        # 第一个贝叶斯线性层
        x_ = self.blinear1(graph_embedding)
        x_ = self.batch_norm1(x_)
        x_ = F.leaky_relu(x_)

        # 中间贝叶斯线性层
        for layer, batch_norm in zip(self.layers, self.batch_norms):
            x_ = layer(x_)
            x_ = batch_norm(x_)
            x_ = F.leaky_relu(x_)

        # 最后一层输出预测值
        output = self.blinear_out(x_)
        prediction = F.sigmoid(output)  # 保证输出非负数

        return prediction

    def predict(self, x, edge_attr, edge_index, configuration_vector, batch_index):
        """在预测时调用"""
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            prediction = self.forward(x, edge_attr, edge_index, configuration_vector, batch_index)
        return prediction


def evaluate_regression(regressor,
                        X,
                        y,
                        samples=50,
                        std_multiplier=2):
    preds = [regressor(*X) for i in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    ic_acc = ic_acc.float().mean()
    return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()


def load_sql_graphs_pyg(csv_path, dbname, user, password, host, port, start_idx=0, end_idx=8877):
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
    print('start load_sql_graphs_pyg')
    # 读取 CSV 数据
    df = pd.read_csv(csv_path).iloc[start_idx:, :]
    print("data shape :", df.shape)
    # df = pd.read_csv(csv_path).iloc[start_idx:, :].sample(n=3200, random_state=42)
    # 设置数据库连接
    database.setup(dbname=dbname, user=user, password=password, host=host, port=port, cache=False)
    sql_graphs_pyg = []
    # 创建index的vector
    attribute_num = 0
    attribute_dict = {}
    for x in df['index'].values:
        for attrs in parse_attribute_config(x):
            for attr in attrs:
                if attr[:2] == 'C ':
                    attr = attr[2:]
                if attr not in attribute_dict:
                    attribute_dict[attr] = attribute_num
                    attribute_num += 1
    #
    # # 保存 attribute_dict 到文件
    # with open('/home/ubuntu/project/mayang/LOGER/core/infer_model/job/attribute_dict.pkl', 'wb') as f:
    #     pickle.dump(attribute_dict, f)
    print("saved attribute_dict.pkl")
    query_list = df['query'].values
    quey_config_list = df['index'].values
    query_plans = df['query_plan_no_index'].values
    for i in range(df.shape[0]):
        # if cnt % 50 == 0:
        #     print(cnt)
        print(i)
        g = dgl_node_and_edge_vectorization(query_list[i], quey_config_list[i], query_plans[i], attribute_dict)
        # 转换为 PyG 数据
        pyg_data = dgl_to_pyg(g, df['improvement'].values[i])
        sql_graphs_pyg.append(pyg_data)

    return sql_graphs_pyg, df['improvement'].values, len(attribute_dict)


def calculate_qerror_list(pred_list, valid_list):
    q_errors = []
    for pred, valid in zip(pred_list, valid_list):
        pred = pred + 1e-3  # 避免除零错误
        valid = valid + 1e-3

        q_error = max(pred / valid, valid / pred)
        q_errors.append(q_error)
    # 计算均值
    mean_q_error = np.mean(q_errors)

    # 计算 90% 和 95% 分位数
    percentile_90 = np.percentile(q_errors, 90)
    percentile_95 = np.percentile(q_errors, 95)

    # 输出结果
    print(f"Mean QError: {mean_q_error}")
    print(f"90th Percentile QError: {percentile_90}")
    print(f"95th Percentile QError: {percentile_95}")
    print(f"Max QError: {max(q_errors)}")

    return q_errors

predicte_num=0
join_num=0
def main():
    global predicte_num,join_num
    data_loader_file = '/tmp/test.pkl'
    data_loader = None
    # 检查是否已有保存的DataLoader
    if not os.path.exists(data_loader_file):
        # 读取数据并设置数据库
        # tpcds-10:/home/ubuntu/project/mayang/Classification/process_data/tpcds/tpcds_train_l.csv
        # tpcds-1:/home/ubuntu/project/mayang/Classification/process_data/tpcds_1/tpcds_train_l.csv
        # job: /home/ubuntu/project/mayang/Classification/process_data/job/job_train_8935.csv

        sql_graphs_pyg, label_list, config_vector_size = load_sql_graphs_pyg(
            csv_path='/home/ubuntu/project/mayang/Classification/process_data/tpcds/tpcds_train_l.csv',
            dbname='indexselection_tpcds___10',
            user='postgres',
            password='password',
            host='127.0.0.1',
            port='5432',
            start_idx=0
        )

        print('len(label_list) 需要是batch_size的倍数:', len(label_list))

        # 创建 PyG DataLoader
        data_loader = DataLoader(sql_graphs_pyg, batch_size=64, shuffle=True)

        # 保存DataLoader到磁盘
        with open(data_loader_file, 'wb') as f:
            pickle.dump(data_loader, f)

        print('Data loaded and saved to disk.')
    else:
        # 从磁盘加载DataLoader
        with open(data_loader_file, 'rb') as f:
            data_loader = pickle.load(f)

        print('DataLoader loaded from disk.')
    print("predicte num :",predicte_num)
    print("join num :",join_num)
    # 初始化模型
    model = ImprovementPredictionModelGNN(
        data_loader.dataset[0].x.shape[1],
        data_loader.dataset[0].edge_attr.shape[1],
        graph_embedding_size=32,
        config_vector_size=data_loader.dataset[0].configuration_vector.shape[0]
    )
    print("paramter :",data_loader.dataset[0].x.shape[1],
        data_loader.dataset[0].edge_attr.shape[1],
        data_loader.dataset[0].configuration_vector.shape[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()  # 均方误差损失函数
    # 初始化学习率调度器
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # 每10个epoch将学习率减半

    model.train()

    print('start DeepApproximateMLL')
    iteration = 0
    mean_qerror_list = []
    for epoch in range(50):
        total_loss = 0
        for batch_idx, batch in enumerate(data_loader):
            optimizer.zero_grad()

            target = torch.clamp(batch.y.view(-1, 1),min=0)

            inputs = (batch.x, batch.edge_attr, batch.edge_index, batch.configuration_vector, batch.batch)

            loss = model.sample_elbo(
                inputs=inputs,
                labels=target,
                criterion=criterion,
                sample_nbr=10,
                complexity_cost_weight=0.01 / len(data_loader)
            )

            iteration += 1
            if iteration % 100 == 0:
                ic_acc, under_ci_upper, over_ci_lower = evaluate_regression(model,
                                                                            inputs,
                                                                            target,
                                                                            samples=25,
                                                                            std_multiplier=3)

                print("CI acc: {:.2f}, CI upper acc: {:.2f}, CI lower acc: {:.2f}".format(ic_acc, under_ci_upper,
                                                                                          over_ci_lower))
                print("Loss: {:.4f}".format(loss))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print("Epoch %d batch_idx%d loss:" % (epoch + 1, batch_idx), loss.item())

        print(f'Epoch {epoch + 1}, Loss: {total_loss:.4f}')

        if (epoch + 1) % 20 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5  # 将学习率减半

        # 随机抽取 20 个样本进行预测
        sample_indices = random.sample(range(len(data_loader.dataset)), 50)
        min_diff = []
        for idx in sample_indices:
            sample_batch = data_loader.dataset[idx]
            sample_x = sample_batch.x
            sample_edge_attr = sample_batch.edge_attr
            sample_edge_index = sample_batch.edge_index
            sample_configuration_vector = sample_batch.configuration_vector

            # 获取对应的 target 值
            target_value = torch.clamp(sample_batch.y,min=0)
            model.eval()
            with torch.no_grad():
                pred_mean = model.predict(sample_x, sample_edge_attr, sample_edge_index, sample_configuration_vector,
                                          sample_batch.batch)
            print(
                f'Sample Index: {idx}, Predicted Mean: {pred_mean}, Target Value: {target_value}, difference: {pred_mean - target_value}')
            if abs(pred_mean - target_value) < 0.1:
                min_diff.append(pred_mean - target_value)

        print('the number of samples < 0.1:', len(min_diff))

        if (epoch + 1) % 2 == 0:
            print('*************** Testing *****************')
            diff_list = []
            pred_list = []
            label_list = []
            for idx in range(len(data_loader.dataset)):
                sample_batch = data_loader.dataset[idx]
                sample_x = sample_batch.x
                sample_edge_attr = sample_batch.edge_attr
                sample_edge_index = sample_batch.edge_index
                sample_configuration_vector = sample_batch.configuration_vector

                # 获取对应的 target 值
                target_value = sample_batch.y.item()
                if target_value<0:
                    target_value=0
                label_list.append(target_value)
                # print(target_value)
                model.eval()
                with torch.no_grad():
                    pred_mean = model.predict(sample_x, sample_edge_attr, sample_edge_index,
                                              sample_configuration_vector,
                                              sample_batch.batch).item()
                    pred_list.append(pred_mean)
                # print(
                #     f'Sample Index: {idx}, Predicted Mean: {pred_mean}, Target Value: {target_value}, difference: {pred_mean - target_value}')
                diff_list.append(pred_mean - target_value)

            q_errors = calculate_qerror_list(pred_list, label_list)
            q_errors_mean=np.mean(q_errors)
            mean_qerror_list.append(q_errors_mean)
            print("mean_qerror_list ：",mean_qerror_list)
            print(q_errors[:10])
            less_1=[x for x in diff_list if abs(x)<0.1]
            print(f'total distance number: {len(less_1)} / {len(diff_list)}, ratio: {100*len(less_1)/len(diff_list)}')
            # model_save_path = f'/home/ubuntu/project/mayang/LOGER/core/infer_model/job/abla/trained_model_epoch_{epoch + 1}.pth'  # 选择保存的路径和文件名
            # torch.save(model.state_dict(), model_save_path)
            # print(f'Model saved to {model_save_path}')


if __name__ == "__main__":
    main()
