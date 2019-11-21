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

dog_dict = np.load('dog_features.npz')
cat_dict = np.load('cat_features.npz')
dog_files, dog_features = dog_dict['image_names'], dog_dict['dog_features']
cat_files, cat_features = cat_dict['image_names'], cat_dict['cat_features']
files = np.concatenate((dog_files, cat_files))
print(dog_features.shape, cat_features.shape)
features = np.concatenate((dog_features, cat_features), axis=0)
labels, centroids = cluster(features)
print(labels)
print(files)

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