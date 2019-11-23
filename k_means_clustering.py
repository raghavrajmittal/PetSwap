import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

def cluster(features):
    # kmeans = KMeans(n_clusters=3)
    kmeans = MiniBatchKMeans(n_clusters=3)
    kmeans.fit(features)

    labels = kmeans.predict(features)
    centroids = kmeans.cluster_centers_
    return labels, centroids

def hypertune(features):
    clusters = range(7, 25)
    losses = []
    for k in clusters:
        kmeans = MiniBatchKMeans(n_clusters=k)
        kmeans.fit(features)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        dist = pairwise_distance(centers, features)
        K, D = centers.shape
        N, D = points.shape
        dist = dist[cluster_idx, np.arange(N)]
        losses.append(np.sum(dist))
    plt.plot(clusters, losses)
    plt.show()

def pairwise_distance(x, y):
    diffs = np.power(x[:, np.newaxis] - y[np.newaxis], 2)
    sum_mat = np.sum(diffs, axis=2)
    dist = np.sqrt(sum_mat)
    return dist


if __name__ == '__main__':
    dog_dict = np.load('dog_features.npz')
    cat_dict = np.load('cat_features.npz')
    dog_files, dog_features = dog_dict['image_names'], dog_dict['dog_features']
    cat_files, cat_features = cat_dict['image_names'], cat_dict['cat_features']
    files = np.concatenate((dog_files, cat_files))
    print(dog_features.shape, cat_features.shape)
    features = np.concatenate((dog_features, cat_features), axis=0)
    hyptertune(features)

'''
# replace with actual data
df = pd.DataFrame(
    {
        "x": [
            12,
            20,
            28,
            18,
            29,
            33,
            24,
            45,
            45,
            52,
            51,
            52,
            55,
            53,
            55,
            61,
            64,
            69,
            72,
        ],
        "y": [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24],
    }
)
fig = plt.figure(figsize=(5, 5))

plt.scatter(list(df["x"]), list(df["y"]))
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid)

plt.xlim(0, max(df["x"]))
plt.ylim(0, max(df["y"]))
plt.show()
'''