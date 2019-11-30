import pickle
from feature_representation import feature_extraction
import numpy as np
from k_means_clustering import pairwise_distance

def find_closest_image(image_path, type):
    features = feature_extraction(image_path)
    with open('kmeans.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    label = kmeans.predict(features)
    labels = np.load('labels.npy')
    cluster_data = np.load('cluster_data.npz')
    files = cluster_data['image_names']
    raw_features = cluster_data['features']
    indices = np.where(labels == label)[0]
    files = files[indices]
    raw_features = raw_features[indices,:]
    dist = np.squeeze(pairwise_distance(raw_features, features))
    sorted_indices = np.argsort(dist)
    files = files[sorted_indices]
    for file in files:
        if file[:4] != type:
            return file

print(find_closest_image('dogs/test/n02116738_9924.jpg', 'dogs'))