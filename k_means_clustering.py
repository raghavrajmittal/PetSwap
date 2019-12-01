import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import pickle

def cluster(features):
    # kmeans = KMeans(n_clusters=3)
    kmeans = MiniBatchKMeans(n_clusters=440)
    kmeans.fit(features)
    labels = kmeans.predict(features)
    centroids = kmeans.cluster_centers_
    with open('kmeans.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    np.save('labels.npy', labels)
    np.save('centroids.npy', centroids)
    #return labels, centroids

def hypertune(features):
    clusters = range(20, 1000, 10)
    losses = []
    for k in clusters:
        kmeans = MiniBatchKMeans(n_clusters=k)
        kmeans.fit(features)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        dist = pairwise_distance(centers, features)
        K, D = centers.shape
        N, D = features.shape
        dist = dist[labels, np.arange(N)]
        losses.append(np.sum(dist))
    np.save('clusters.npy', clusters)
    np.save('losses.npy', losses)
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
    dog_files, dog_features = dog_dict['image_names'], dog_dict['features']
    cat_files, cat_features = cat_dict['image_names'], cat_dict['features']
    files = np.concatenate((dog_files, cat_files))
    features = np.squeeze(np.concatenate((dog_features, cat_features), axis=0))
    indices = np.unique(np.where(features==np.inf)[0])
    features = np.delete(features, indices, axis=0)
    files = np.delete(files, indices)
    np.savez('cluster_data.npz', image_names=files, features=features)
    cluster(features)
