import numpy as np
import pickle
from kmodes.kmodes import KModes

def cluster(features):
    model = KModes(n_clusters=15, init='Huang', n_init=20)
    kmodes_labels = model.fit_predict(features)
    np.save('kmodes_labels.npy', kmodes_labels)
    with open('kmodes.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    cluster_data = np.load('cluster_data.npz')
    files = cluster_data['image_names']
    features = cluster_data['features']
    cluster(features)