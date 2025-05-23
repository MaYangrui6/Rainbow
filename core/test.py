import pandas as pd
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.loader import DataLoader
import gpytorch
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood
import os
import sys
import random

loger_dir = os.path.abspath('/home/ubuntu/project/mayang/LOGER')

# 将 QPE 目录添加到 sys.path
sys.path.append(loger_dir)
from core.models.DGP import DeepGPModel
from core.models.DKL import DKLGPModel
from sql import Sql, _global
from core import database, Sql, Plan, load
import ast
import re
import torch.nn as nn
import torch.nn.functional as F

# 获取 QPE 目录的绝对路径
qpe_dir = os.path.abspath('/home/ubuntu/project/mayang')

# 将 QPE 目录添加到 sys.path
sys.path.append(qpe_dir)

# 现在尝试导入 TreeBuilder 类
from QPE.sql2fea import TreeBuilder


def contains_number(s: str) -> bool:
    """
    判断字符串中是否包含数字。

    :param s: 输入字符串
    :return: 如果包含数字返回 True，否则返回 False
    """
    return bool(re.search(r'\d', s))


def convert_numeric_comparison(expression: str) -> str:
    """
    将输入字符串中的数字转化为 '数字'::numeric 格式。
    例如：catalog_sales.cs_sales_price > 500 -> catalog_sales.cs_sales_price > '500'::numeric
    """

    # 定义正则表达式来匹配比较运算符两边的数字 (包括小数) 和运算符号
    pattern = re.compile(r"([<>=!]+)\s*(\d+(\.\d+)?)")

    # 替换匹配到的数字，并将其转化为 '数字'::numeric 格式
    def replacer(match):
        operator = match.group(1)
        number = match.group(2)
        return f"{operator} '{number}'::numeric"

    # 使用正则表达式替换原始字符串中的数字
    converted_expression = pattern.sub(replacer, expression)

    return expression.split('.')[0], converted_expression

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
            if '=' not in part:
                current_num = 1
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
        print('The input string is not a valid BETWEEN expression. cmp',cmp)
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
            raise ValueError("在 IN 表达式中未找到有效值。")

        # 单个值的情况：转化为等值比较
        if len(value_list) == 1:
            new_expression = f"(({column})::text = '{value_list[0]}'::text)"
        else:
            # 多个值的情况：转化为 ANY 比较
            formatted_values = ','.join(f"'{v}'" for v in value_list)
            new_expression = f"(({column})::text = ANY ('{{{formatted_values}}}'::text[]))"

        return table_name, new_expression

    # 无法匹配时，抛出错误
    # print(f"无法匹配的输入: {cmp}")  # 调试信息
    return cmp.split('.')[:2]
    # raise ValueError("输入字符串不是有效的比较表达式。")


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


def delete_left_table_cond(sql_condition):
    # 使用正则表达式匹配表名和列名
    match = re.match(r"[\w\.]+\.(\w+)\s*(=|<>|<|>|<=|>=)\s*(.*)", sql_condition)

    if not match:
        # print('sql_condition', sql_condition)
        # raise ValueError("The input string is not a valid comparison expression.")
        return 'not match in function delete_left_table_cond'

    # 提取列名、运算符和右侧表达式
    column = match.group(1)
    operator = match.group(2)
    right_side = match.group(3)

    # 返回去掉表名前缀的表达式
    return f"{column} {operator} {right_side.strip()}"


def dgl_node_and_edge_vectorization(sql_query, config_index, plan, attribute_dict):
    sql_instance = Sql(sql_query)
    # 构建异构图
    g, data_dict, node_indexes, edge_lists = sql_instance.to_hetero_graph_dgl()
    #处理edge_list中一个语句存在很多and、or语句的情况
    edge_list=[]
    for cmp,num_table in edge_lists:
        if 'BETWEEN' in cmp:
            edge_list += [(cmp, num_table)]
        elif 'OR' in cmp or 'AND' in cmp:
            cmp_split=split_cmp_expression(cmp,num_table)
            edge_list += cmp_split
        else:
            edge_list += [(cmp,num_table)]
    # 构建 node feature
    filter_features = g.ndata['filter']
    edge_features = g.ndata['edge']
    onehot_features = g.ndata['onehot']
    # 获取其他特征
    others_features = g.ndata['others']

    # 检查filter特征的大小
    filter_shape = g.ndata['filter'].shape[0]

    # 如果others_features为空，创建一个对应于filter特征的空张量
    if others_features.numel() == 0:
        others_features = torch.zeros(filter_shape, 12 * database.schema.max_columns)

    # 检查filter_shape是否为0，避免重塑操作
    if filter_shape > 0:
        # 现在可以安全地重塑others_features
        others_features = others_features.view(filter_shape, -1)

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
            # elif contains_number(cmp):
            #     table, cmp_in_op = convert_numeric_comparison(cmp)
            else:
                table, cmp_in_op = cmp.split('.')[:2]

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
        if operator_vectors_cmp2:
            vector_edge = torch.cat(operator_vectors_cmp2, dim=0)
        else:
            vector_edge = torch.zeros(0, num_edge_features)
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


class ImprovementPredictionModelGNN(nn.Module):
    """使用图卷积神经网络进行预测，基于图编码，并增加多层线性层，融合配置向量."""

    def __init__(self, in_feats, edge_feats, graph_embedding_size, config_vector_size, hidden_dim=128, num_layers=3):
        print('start ImprovementPredictionModelGNN')
        super(ImprovementPredictionModelGNN, self).__init__()

        # 图编码器
        self.graph_encoder = GraphEncoder(in_feats, edge_feats, graph_embedding_size)

        # 定义多个全连接层
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()  # 批归一化层列表

        # 第一层，从 (图嵌入维度 + 配置向量维度) 到隐藏层维度
        input_dim = graph_embedding_size * 2 + config_vector_size  # 2 * graph_embedding_size 因为全局池化时合并了 mean 和 max
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))  # 添加批归一化层

        # 中间的隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # 最后一层，隐藏层到输出维度
        self.layers.append(nn.Linear(hidden_dim, 1))  # 输出为单个值

    def forward(self, x, edge_attr, edge_index, configuration_vector, batch_index):
        # 通过图编码器获取图的嵌入表示
        graph_embedding = self.graph_encoder(x, edge_attr, edge_index, batch_index)

        # 拼接 graph_embedding 和 configuration_vector
        configuration_vector = configuration_vector.view(graph_embedding.size(0), -1)  # 调整 configuration_vector 的形状
        graph_embedding = torch.cat([graph_embedding, configuration_vector], dim=1)

        # 将拼接后的向量输入到多层全连接网络
        for i, layer in enumerate(self.layers[:-1]):
            graph_embedding = layer(graph_embedding)  # 全连接层
            graph_embedding = self.batch_norms[i](graph_embedding)  # 批归一化层
            graph_embedding = F.relu(graph_embedding)  # ReLU 激活函数

        # 最后一层输出预测值
        output = self.layers[-1](graph_embedding)
        prediction = F.sigmoid(output)  # 使用 ReLU 激活函数保证输出非负数

        return prediction

    def predict(self, x, edge_attr, edge_index, configuration_vector, batch_index):
        """在预测时调用"""
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            graph_embedding = self.graph_encoder(x, edge_attr, edge_index, batch_index)

            # 拼接 graph_embedding 和 configuration_vector
            configuration_vector = configuration_vector.view(graph_embedding.size(0), -1)
            graph_embedding = torch.cat([graph_embedding, configuration_vector], dim=1)

            # 通过多层全连接网络计算预测值
            for i, layer in enumerate(self.layers[:-1]):
                graph_embedding = layer(graph_embedding)
                graph_embedding = self.batch_norms[i](graph_embedding)
                graph_embedding = F.relu(graph_embedding)

            output = self.layers[-1](graph_embedding)
            prediction = F.sigmoid(output)  # 保证预测值为非负数

        return prediction


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

num_edge_features = 15

# 设置随机种子
seed = 47
torch.manual_seed(seed)  # PyTorch 随机种子
np.random.seed(seed)  # NumPy 随机种子
random.seed(seed)  # Python 随机种子

import pandas as pd
df=pd.read_csv('/home/ubuntu/project/mayang/Classification/process_data/tpcds/tpcds_plan_gene_f.csv').iloc[:,1:]
print(df.shape)
database.setup(dbname='indexselection_tpcds___1', user='postgres', password='password', host='127.0.0.1', port='5432', cache=False)
i=1145
dgl_node_and_edge_vectorization(df['query'].values[i], df['index'].values[i], df['plan'].values[i], {})