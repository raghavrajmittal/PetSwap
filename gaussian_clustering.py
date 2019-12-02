from sklearn.mixture import GaussianMixture
import numpy as np
import pickle

def cluster(features):
    model = GaussianMixture(n_components=15).fit(features)
    gaussian_labels = model.predict(features)
    np.save('gaussian_labels.npy', gaussian_labels)
    with open('gaussian.pkl', 'wb') as f:
        pickle.dump(model, f)


def hypertune(features):
    components = range(1, 50)
    aics = []
    bics = []
    for i in components:
        model = GaussianMixture(n_components=i).fit(features)
        aics.append(model.aic(features))
        bics.append(model.bic(features))
    np.save('aics.npy', aics)
    np.save('bics.npy', bics)



if __name__ == '__main__':
    cluster_data = np.load('cluster_data.npz')
    files = cluster_data['image_names']
    features = cluster_data['features']
    cluster(features)