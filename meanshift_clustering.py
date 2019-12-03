import numpy as np
import pickle
from sklearn.cluster import MeanShift

def cluster(features):
    model = MeanShift().fit(features)
    meanshift_labels = model.predict(features)
    print(meanshift_labels)
    np.save('meanshift_labels.npy', meanshift_labels)
    with open('meanshift.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    cluster_data = np.load('cluster_data.npz')
    files = cluster_data['image_names']
    features = cluster_data['features']
    cluster(features)