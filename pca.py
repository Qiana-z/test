import numpy as np
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt

def pca_reduce_two(pos_path, neg_path, output_dir="ml_data", target_dim=1024):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取数据
    X_pos = np.load(pos_path)   # (N1, 2048)
    X_neg = np.load(neg_path)   # (N2, 2048)

    # 2. 合并数据（一起做PCA，保证同一投影空间）
    X = np.vstack([X_pos, X_neg])
    print(f"原始数据: {X.shape[0]} 样本, {X.shape[1]} 维")

    # 3. 训练 PCA
    pca = PCA(n_components=target_dim, random_state=42)
    X_reduced = pca.fit_transform(X)
    print(f"降维后: {X_reduced.shape[1]} 维, 解释方差比例={pca.explained_variance_ratio_.sum():.4f}")

    # 4. 拆分回 pos/neg
    X_pos_reduced = X_reduced[:X_pos.shape[0]]
    X_neg_reduced = X_reduced[X_pos.shape[0]:]

    # 5. 保存结果
    np.save(os.path.join(output_dir, f"pos_pca_{target_dim}.npy"), X_pos_reduced)
    np.save(os.path.join(output_dir, f"neg_pca_{target_dim}.npy"), X_neg_reduced)

    print(f"✅ 已保存到 {output_dir}")
    return X_pos_reduced, X_neg_reduced, pca

def pca_reduce(emb_path, output_dir="ml_data", target_dim=1024):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取数据
    X = np.load(emb_path)   # (N1, 2048)
    print(f"原始数据: {X.shape[0]} 样本, {X.shape[1]} 维")

    # 3. 训练 PCA
    pca = PCA(n_components=target_dim, random_state=42)
    X_reduced = pca.fit_transform(X)
    print(f"降维后: {X_reduced.shape[1]} 维, 解释方差比例={pca.explained_variance_ratio_.sum():.4f}")

    # 5. 保存结果
    np.save(os.path.join(output_dir, f"test_pca_{target_dim}.npy"), X_reduced)
    print(f"✅ 已保存到 {output_dir}")

    return X_reduced, pca

# pos_reduced, neg_reduced, pca_model = pca_reduce_two(
#     pos_path="ml_data/embeddings_pos.npy",
#     neg_path="ml_data/embeddings_neg.npy",
#     output_dir="ml_data",
#     target_dim=1024
# )

pca_reduce(
    emb_path="ml_data/embeddings_test.npy",
    output_dir="ml_data",
    target_dim=1024
)

# ====== 读取正负例数据 ======
# X_pos = np.load("ml_data/pos_pca_1024.npy")
# X_neg = np.load("ml_data/neg_pca_1024.npy")

# # 打标签：正类=1，负类=0
# y_pos = np.ones(X_pos.shape[0], dtype=int)
# y_neg = np.zeros(X_neg.shape[0], dtype=int)

# # 合并
# X = np.vstack([X_pos, X_neg])
# y = np.concatenate([y_pos, y_neg])

# print(f"总样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")

# # ====== 4. PCA 方差解释 ======
# pca_full = PCA()
# pca_full.fit(X)

# explained_variance_ratio = pca_full.explained_variance_ratio_
# cumulative_variance = explained_variance_ratio.cumsum()

# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
# plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
# plt.xlabel('Number of Principal Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('PCA Explained Variance')
# plt.legend()
# plt.grid(True)
# plt.show()