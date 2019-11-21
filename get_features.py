'''save training data in an npz file'''

from feature_representation import feature_extraction
import glob
import numpy as np
import skimage.io


# get features given a dataset
def get_features(dir):
    fnames = glob.glob(dir + '*.jpg')[:6]
    files = []
    featureArr = None
    for fname in fnames:
        print(fname)
        im = skimage.io.imread(fname)
        features = feature_extraction(im)
        if features is not None:
            files.append(fname)
            if featureArr is None:
                featureArr = features
            else:
                featureArr = np.append(featureArr, features, axis=0)
        print(featureArr.shape)
    return files, featureArr

if __name__ == "__main__":
    dogs_train_dir = 'dogs/test/'
    fnames, dog_features = get_features(dogs_train_dir)
    np.savez('dog_features.npz', image_names=fnames, dog_features=dog_features)

    cats_train_dir = 'cats/test/'
    fnames, cat_features = get_features(cats_train_dir)
    np.savez('cat_features.npz', image_names=fnames, cat_features=cat_features)
