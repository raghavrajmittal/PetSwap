import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops


def feature_extraction(img, mask):
    # Get pixels that are masked
    # Get color representation
    # Get texture representation
    # Fuse them


def get_color_features(img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    # rs = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    color_features = np.array(
        [
            np.mean(red),
            np.std(red),
            np.mean(green),
            np.std(green),
            np.mean(blue),
            np.std(blue)
        ]
    )
    return color_features


def get_hsv_features(img, mask):
    pass


def get_glcm_features(img, mask):
    # https://stackoverflow.com/questions/47591359/calculating-the-co-occurrence-matrix-of-part-of-an-image-from-a-mask
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = feature.greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    return greycoprops(glcm)

def get_gabor_features(img, mask):
    pass