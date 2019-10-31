import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# replace with actual data
df = pd.DataFrame({
    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
})

kmeans = KMeans(n_clusters=3)
kmeans.fit(df)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(5, 5))

plt.scatter(list(df['x']), list(df['y']))
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid)

plt.xlim(0, max(df['x']))
plt.ylim(0, max(df['y']))
plt.show()