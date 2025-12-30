import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 

# # ====== 1. 读取测试集数据 ======
# X_test = np.load("ml_data/test_pca_1024.npy")  # 测试集的向量
# df = pd.read_csv("ml_data/embeddings_test_with_index.csv")

# # 标签列：是否有风险（正例=1，负例=0）
# if "是否有风险" not in df.columns:
#     raise KeyError("CSV 文件中未找到列: 是否有风险")

# y_test = df["是否有风险"].astype(int).values

# # 检查维度一致
# if X_test.shape[0] != len(y_test):
#     raise ValueError(f"向量数 {X_test.shape[0]} 与 标签数 {len(y_test)} 不一致")

# print(f"测试集样本数: {X_test.shape[0]}, 特征维度: {X_test.shape[1]}")

# # ====== 2. t-SNE 三维降维 ======
# tsne = TSNE(n_components=3, random_state=42, perplexity=30)
# X_tsne = tsne.fit_transform(X_test)

# # ====== 3. 绘制 3D 可视化 ======
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# scatter = ax.scatter(
#     X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2],
#     c=y_test, cmap="viridis", alpha=0.6, s=20
# )

# legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
# ax.add_artist(legend1)

# ax.set_title("3D t-SNE Visualization of Test Embeddings")
# ax.set_xlabel("Dim 1")
# ax.set_ylabel("Dim 2")
# ax.set_zlabel("Dim 3")

# plt.show()

# ====== 2. t-SNE 可视化 ======
# tsne = TSNE(n_components=2, random_state=42, perplexity=30)
# X_tsne = tsne.fit_transform(X_test)

# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(
#     X_tsne[:, 0], X_tsne[:, 1],
#     c=y_test, cmap="viridis", alpha=0.6
# )
# plt.legend(*scatter.legend_elements(), title="Classes")
# plt.title("t-SNE Visualization of Test Embeddings")
# plt.show()

# # ====== 1. 读取正负例数据 ======
X_pos = np.load("ml_data/pos_pca_1024.npy")
X_neg = np.load("ml_data/neg_pca_1024.npy")

# 打标签：正类=1，负类=0
y_pos = np.ones(X_pos.shape[0], dtype=int)
y_neg = np.zeros(X_neg.shape[0], dtype=int)

# 合并
X = np.vstack([X_pos, X_neg])
y = np.concatenate([y_pos, y_neg])

print(f"总样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")

# ====== 2. t-SNE 三维可视化 ======
tsne = TSNE(n_components=3, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2],
    c=y, cmap="viridis", alpha=0.6, s=20
)

legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)

ax.set_title("3D t-SNE Visualization of Embeddings")
ax.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")
ax.set_zlabel("Dim 3")

plt.show()

# # ====== 2. t-SNE 可视化 ======
# tsne = TSNE(n_components=2, random_state=42, perplexity=30)
# X_tsne = tsne.fit_transform(X)

# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
# plt.legend(*scatter.legend_elements(), title="Classes")
# plt.title('t-SNE Visualization of Embeddings')
# plt.show()

# ====== 3. PCA 可视化 ======
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
# plt.legend(*scatter.legend_elements(), title="Classes")
# plt.xlabel('First Principal Component')
# plt.ylabel('Second Principal Component')
# plt.title('PCA Visualization of Embeddings')
# plt.show()

# ====== 4. PCA 方差解释率 ======
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