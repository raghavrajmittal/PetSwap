'''get training data in the form of arr[tuple1, tuple2, ...]
where tuple is: (path_to_image, 'cat'/'dog', features)'''

from feature_representation import feature_extraction
import glob
import skimage.io

cats_train_dir = 'cats/train/'
dogs_train_dir = 'dogs/train/'

# get features for cats
def get_training_features_cat():
    fnames = glob.glob(cats_train_dir + '*.jpg')
    featureArr = []
    for fname in fnames:
        im = skimage.io.imread(fname)
        features = feature_extraction(img)
        featureArr.append((fname, "cat", features))
        return featureArr

# get features for dogs
def get_training_features_dog():
    fnames = glob.glob(dogs_train_dir + '*.jpg')
    featureArr = []
    for fname in fnames:
        im = skimage.io.imread(fname)
        features = feature_extraction(img)
        featureArr.append((fname, "dog", features))
        return featureArr
