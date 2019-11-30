import numpy as np
import glob

folder = 'dog_features'
zipFiles = np.sort(glob.glob(folder + '/*.npz'))
allFiles = []
allFeatures = []
for zipFile in zipFiles:
    npobject = np.load(zipFile)
    images = npobject['image_names']
    features = npobject['features']
    for idx in range(len(images)):
        allFiles.append(images[idx])
        allFeatures.append(features[idx])
np.savez(folder + '.npz', image_names=np.array(allFiles), features=np.array(allFeatures))