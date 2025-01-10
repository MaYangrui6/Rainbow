import ast
import numpy as np
from gensim.models import KeyedVectors
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import WordEmbeddingSimilarityIndex, SparseTermSimilarityMatrix, SoftCosineSimilarity
from gensim.similarities.annoy import AnnoyIndexer

from HyperQO.sql_feature.bag_of_predicates import BagOfPredicates


def embed_queries_and_plans(sql_embedder, workload, workload_plans):
    embeddings = sql_embedder.get_embedding(workload)
    bag = BagOfPredicates()
    predicates = []
    for plan in workload_plans:
        predicate = bag.extract_predicates_from_plan(plan["Plan"])
        predicates.append(predicate)
    dictionary = Dictionary(predicates)
    return embeddings, predicates, dictionary


# def build_similarity_index(model, embeddings, predicates, dictionary, num_trees=40, num_best=10):
#     indexer = AnnoyIndexer(model, num_trees=num_trees)
#     tfidf = TfidfModel(dictionary=dictionary)
#     termsim_index = WordEmbeddingSimilarityIndex(embeddings, kwargs={'indexer': indexer})
#     similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)
#     tfidf_corpus = tfidf[[dictionary.doc2bow(predicate) for predicate in predicates]]
#     docsim_index = SoftCosineSimilarity(tfidf_corpus, similarity_matrix, num_best=num_best)
#     return docsim_index
def build_similarity_index(model, embeddings, predicates, dictionary, num_trees=40, num_best=10):
    # 确认模型类型
    if isinstance(model, KeyedVectors):
        print("Model is KeyedVectors")
    else:
        print("Model is not KeyedVectors, converting...")
        model = model.wv

    # 确认 embeddings 格式
    if not hasattr(embeddings, 'shape'):
        print("Converting embeddings to numpy array")
        embeddings = np.array(embeddings)

    # 确认字典大小
    if len(dictionary) <= 1:
        print("Dictionary size is too small, check your dictionary construction process")
        # 重新构建字典（假设 predicates 是一个列表，包含所有的文本数据）
        dictionary = Dictionary(predicates)

    # 创建索引器
    indexer = AnnoyIndexer(model, num_trees=num_trees)

    # 创建 TF-IDF 模型
    tfidf = TfidfModel(dictionary=dictionary)

    # 创建词嵌入相似度索引
    termsim_index = WordEmbeddingSimilarityIndex(embeddings, kwargs={'indexer': indexer})

    # 创建稀疏术语相似度矩阵
    similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)

    # 将谓词转换为 TF-IDF 语料库
    tfidf_corpus = tfidf[[dictionary.doc2bow(predicate) for predicate in predicates]]

    # 创建软余弦相似度索引
    docsim_index = SoftCosineSimilarity(tfidf_corpus, similarity_matrix, num_best=num_best)

    return docsim_index
