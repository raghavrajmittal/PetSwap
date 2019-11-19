'''get training data in the form of arr[tuple1, tuple2, ...]
where tuple is: (path_to_image, 'cat'/'dog', features)'''

from feature_representation import feature_extraction
import glob
import numpy as np
import skimage.io


# get features for dogs
def get_features(dir, type):
    fnames = glob.glob(dir + '*.jpg')
    featureArr = []
    for fname in fnames:
        im = skimage.io.imread(fname)
        features = feature_extraction(im)
        featureArr.append((fname, type, features))
        return featureArr

if __name__ == "__main__":
    cats_train_dir = 'cats/train/'
    dogs_train_dir = 'dogs/train/'
    dog_features = get_features(dogs_train_dir, 'dog')
    cat_features = get_features(cats_train_dir, 'cat')
    np.savez('features.npz', dog_features=dog_features, cat_features=cat_features)