import torch


def initialize_inducing_points(x_train, method='kmeans', k=128):
    """
    引入诱导点来近似真实的训练数据，这可以部分减轻 DGP 的计算复杂度，我们这里更偏向于 kmeans 的方法
    用于从输入数据初始化诱导点的方法。
    :param x_train: 训练数据，用于初始化诱导点
    :param method: 诱导点初始化方法, 'random' 或 'kmeans'
    :param k: 诱导点的数量
    :return: 根据训练数据初始化的诱导点
    """
    if method == 'random':
        # 从训练数据中随机采样作为诱导点
        indices = torch.randperm(x_train.size(0))[:k]
        return x_train[indices].clone().detach()
    elif method == 'kmeans':
        # 使用 K-means 聚类来找到 k 个中心点作为诱导点
        # 能够在一定程度上加速模型的收敛，避免由于诱导点过远而导致的训练困难问题
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k).fit(x_train.cpu().numpy())
        return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown method {method} for initializing inducing points.")