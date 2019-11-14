'''helper class to get feature representation of a given image'''
import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage import io
import maskrcnn


def feature_extraction(img):
    # Get pixels that are masked
    results = maskrcnn.get_masks(img)
    mask = results[0]["masks"]
    # Get color representation
    color_features = get_color_features(img, mask)
    hist_features = get_hsv_features(img, mask)
    # Get texture representation
    glcm_features = get_glcm_features(img, mask)
    # Fuse them
    return np.concatenate((color_features, hist_features, glcm_features[0]), axis=None)


def get_color_features(img, mask):
    other = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(float)
    other = other.astype(float)
    mask = np.tile(mask, (1, 1, 3))
    img[~mask] = np.NaN
    other[~mask] = np.NaN
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    color_features = np.array(
        [
            np.nanmean(red),
            np.nanstd(red),
            np.nanmean(green),
            np.nanstd(green),
            np.nanmean(blue),
            np.nanstd(blue),
            np.nanmean(other),
            np.nanstd(other),
        ]
    )
    return color_features


def get_hsv_features(img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.tile(mask, (1, 1, 3))
    hist_mask = np.zeros(mask.shape).astype(np.uint8)
    hist_mask[~mask] = 0
    hist_mask[mask] = 255
    h = cv2.calcHist(
        [img], channels=[0], mask=hist_mask[:, :, 0], histSize=[8], ranges=[0, 8]
    ) / np.max(img[:, :, 0])
    s = cv2.calcHist(
        [img], channels=[1], mask=hist_mask[:, :, 1], histSize=[2], ranges=[0, 2]
    ) / np.max(img[:, :, 1])
    v = cv2.calcHist(
        [img], channels=[2], mask=hist_mask[:, :, 2], histSize=[2], ranges=[0, 2]
    ) / np.max(img[:, :, 2])
    return np.concatenate((h, s, v), axis=None)


def get_glcm_features(img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.squeeze(mask)
    img[~mask] = -1
    glcm = greycomatrix(
        img, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True
    )[1:, 1:, :, :]
    return greycoprops(glcm)


def get_gabor_features(img, mask):
    pass


# print(feature_extraction(io.imread("test_cat.jpg")))
