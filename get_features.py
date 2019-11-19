'''save training data in an npz file'''

from feature_representation import feature_extraction
import glob
import numpy as np
import skimage.io


# get features given a dataset
def get_features(dir, type):
    fnames = glob.glob(dir + '*.jpg')
    featureArr = None
    for fname in fnames:
        print(fname)
        im = skimage.io.imread(fname)
        features = feature_extraction(im)
        if featureArr is None:
            featureArr = features
        else:
            featureArr = np.concatenate((featureArr, features), axis=0)
    return fnames, featureArr

if __name__ == "__main__":
    dogs_train_dir = 'dogs/train/'
    fnames, dog_features = get_features(dogs_train_dir, 'dog')
    np.savez('dog_features.npz', image_names=fnames, dog_features=dog_features)

    cats_train_dir = 'cats/train/'
    fnames, cat_features = get_features(cats_train_dir, 'cat')
    np.savez('cat_features.npz', image_names=fnames, cat_features=cat_features)
