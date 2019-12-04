import pickle
from feature_representation import feature_extraction
import numpy as np
from k_means_clustering import pairwise_distance

def find_closest_image(image_path, type, clustering='GMM'):
    features = feature_extraction(image_path)
    if clustering == 'kmeans':
        with open('kmeans.pkl', 'rb') as f:
            model = pickle.load(f)
            labels = np.load('labels.npy')
            label = model.predict(features)
    elif clustering == 'GMM':
        with open('gaussian.pkl', 'rb') as f:
            model = pickle.load(f)
            labels = np.load('gaussian_labels.npy')
            label = model.predict(features)
    elif clustering == 'meanshift':
        with open('meanshift.pkl', 'rb') as f:
            model = pickle.load(f)
            labels = np.load('meanshift_labels.npy')
            label = model.predict(features)
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

img_path = 'dogs/test/n02102040_1178.jpg'
kmeansimg = find_closest_image(img_path, 'dogs', 'kmeans')
meanshiftimg = find_closest_image(img_path, 'dogs', 'meanshift')
GMMimg = find_closest_image(img_path, 'dogs', 'GMM')
print("test image: ", img_path)
print("kmeans image: ", kmeansimg)
print("meanshift image: ", meanshiftimg)
print("GMM image: ", GMMimg)