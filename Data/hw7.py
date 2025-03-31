import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# --------------------------------------------------
# Part 1: K-Means Clustering 
# --------------------------------------------------

# Load and clean the dataset
data = pd.read_csv("Spotify_Youtube.csv")
x = data[["Liveness", "Energy", "Loudness"]].dropna().values

# Elbow method to find optimal K
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
    kmeans.fit(x)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("SSE")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

# Run KMeans with K=3
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, n_init=10, max_iter=300, random_state=42)
labels = kmeans.fit_predict(x)
centers = kmeans.cluster_centers_

# Organize data points by cluster
clusters = {str(i): [[], [], []] for i in range(num_clusters)}
for i in range(len(x)):
    c = labels[i]
    clusters[str(c)][0].append(x[i][0])
    clusters[str(c)][1].append(x[i][1])
    clusters[str(c)][2].append(x[i][2])

# 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
colors = ['red', 'green', 'blue']

for i in range(num_clusters):
    xs = clusters[str(i)][0]
    ys = clusters[str(i)][1]
    zs = clusters[str(i)][2]
    ax.scatter(xs, ys, zs, c=colors[i % len(colors)], label=f'Cluster {i}', alpha=0.6)

for center in centers:
    ax.scatter(center[0], center[1], center[2], c='black', marker='*', s=200, label='Centroid')

ax.set_xlabel("Liveness")
ax.set_ylabel("Energy")
ax.set_zlabel("Loudness")
ax.set_title("K-Means Clustering")
ax.legend()
plt.show()


# --------------------------------------------------
# Part 2: Hierarchical Clustering 
# --------------------------------------------------

# Standardize the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Create linkage matrix and plot dendrogram
linked = linkage(x_scaled, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

# Cut the dendrogram to form flat clusters
n_clusters_hier = 3
hier_labels = fcluster(linked, n_clusters_hier, criterion='maxclust')

# Print out simple summary
print("Cluster counts (Hierarchical):")
for i in range(1, n_clusters_hier + 1):
    count = list(hier_labels).count(i)
    print(f"Cluster {i}: {count} samples")
