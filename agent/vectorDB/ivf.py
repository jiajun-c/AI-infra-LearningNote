import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
import time
from collections import defaultdict
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'STHeiti', 'PingFang SC', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以保证结果可重现
np.random.seed(42)

def generate_sample_data(n_samples=1000, dim=2):
    """生成示例数据：三个明显分离的高斯分布簇"""
    # 第一个簇
    cluster1 = np.random.normal(loc=[2, 2], scale=0.5, size=(n_samples//3, dim))
    # 第二个簇  
    cluster2 = np.random.normal(loc=[8, 3], scale=0.6, size=(n_samples//3, dim))
    # 第三个簇
    cluster3 = np.random.normal(loc=[5, 8], scale=0.4, size=(n_samples - 2*(n_samples//3), dim))
    
    data = np.vstack([cluster1, cluster2, cluster3])
    return data

# 生成数据
data = generate_sample_data()
print(f"数据形状: {data.shape}")

class SimpleKMeans:
    """简化的K-means实现用于IVF聚类"""
    
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels_ = None
    
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # 1. 随机初始化质心
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        
        for iteration in range(self.max_iters):
            # 2. 分配每个点到最近的质心
            distances = euclidean_distances(X, self.centroids)
            labels = np.argmin(distances, axis=1)
            
            # 3. 更新质心位置
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            
            # 检查收敛
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
        
        self.labels_ = labels
        return self
    
class SimpleIVF:
    """简化的IVF实现"""
    
    def __init__(self, n_clusters=3, n_probe=2):
        self.n_clusters = n_clusters
        self.n_probe = n_probe  # 搜索时探测的簇数量
        self.kmeans = None
        self.inverted_lists = None  # 倒排列表
        self.centroids = None
        self.is_trained = False
        
    def train(self, data):
        print("train ivf index")
        self.kmeans = SimpleKMeans(n_clusters=self.n_clusters)
        self.kmeans.fit(data)
        self.centroids = self.kmeans.centroids
        self.is_trained = True
    
    def build_index(self, data):
        if not self.is_trained:
            self.train(data)
        
        self.inverted_lists = defaultdict(list)
        distance = euclidean_distances(data, self.centroids)
        labels = np.argmin(distance, axis=1)
        for idx, label in enumerate(labels):
            self.inverted_lists[label].append(idx)
        
    # query的形状为(1, nfeature)
    # 返回的形状为 (1, K)
    def search(self, query, k=5, data=None):
        if data is None:
            data = self.data
        distance_to_centroids = euclidean_distances([query], self.centroids)[0]
        nearst_cluster_indices = np.argsort(distance_to_centroids)[:self.n_probe]
        
        candidate_indices = []
        for cluster_idx in nearst_cluster_indices:
            candidate_indices.extend(self.inverted_lists[cluster_idx])
        
        if not candidate_indices:
            return [], []

        candidate_vectors = data[candidate_indices]
        distances = euclidean_distances([query], candidate_vectors)[0]
        
        k = min(k, len(distances))
        nearest_indices_within_candidates = np.argsort(distances)[:k]
        
        # 映射回原始索引
        final_indices = [candidate_indices[i] for i in nearest_indices_within_candidates]
        final_distances = distances[nearest_indices_within_candidates]
        
        return final_indices, final_distances
    
def demonstrate_ivf():
    """完整演示IVF算法"""
    print("=" * 60)
    print("IVF算法演示")
    print("=" * 60)
    
    # 1. 生成数据
    data = generate_sample_data(300, 2)
    print(f"生成{len(data)}个二维数据点")
    
    # 2. 创建并训练IVF索引
    ivf = SimpleIVF(n_clusters=3, n_probe=2)
    ivf.data = data  # 保存数据引用
    ivf.build_index(data)
    
    # 3. 选择一个查询点
    query_point = np.array([5.0, 5.0])
    print(f"\n查询点: {query_point}")
    
    # 4. 使用IVF搜索
    start_time = time.time()
    ivf_indices, ivf_distances = ivf.search(query_point, k=5, data=data)
    ivf_time = time.time() - start_time
    
    # 5. 使用暴力搜索作为对比
    start_time = time.time()
    # bf_indices, bf_distances = ivf.brute_force_search(query_point, k=5, data=data)
    # bf_time = time.time() - start_time
    
    # 6. 显示结果
    print(f"\n搜索结果对比:")
    print(f"IVF搜索  - 找到{len(ivf_indices)}个最近邻, 耗时: {ivf_time:.6f}秒")
    # print(f"暴力搜索 - 找到{len(bf_indices)}个最近邻, 耗时: {bf_time:.6f}秒")
    
    # print(f"\n速度提升: {bf_time/ivf_time:.2f}倍")
    
    print(f"\n最近邻索引 (IVF): {ivf_indices}")
    # print(f"最近邻索引 (暴力): {bf_indices}")
    
    # 7. 检查召回率
    # intersection = set(ivf_indices) & set(bf_indices)
    # recall = len(intersection) / len(bf_indices)
    # print(f"召回率: {recall:.2%} ({len(intersection)}/{len(bf_indices)})")
    
    return ivf, data, query_point, ivf_indices

# 运行演示
ivf, data, query, ivf_results = demonstrate_ivf()