# /usr/bin/python
import numpy as np
import os
import cv2


def get_data(dataset_dir):
    images_list = []
    for d in os.walk(dataset_dir):
        for file in os.listdir(d[0]):
            if file.endswith(".jpg"):
                images_list.append(os.path.join(os.getcwd(), d[0], file))
    return images_list


def split_data(dataset, split_val):
    os.mkdir("train")
    os.mkdir("test")
    i = 0
    while i < len(dataset):
        img = cv2.imread(dataset[i])
        if i < int(split_val * len(dataset)):
            cv2.imwrite(os.path.join("train", dataset[i]), img)
        else:
            cv2.imwrite(os.path.join("test", dataset[i]), img)
        i = i + 1


if __name__ == "__main__":
    dog_path = "./dogs/images/Images"
    # cat_path = "./cats/"
    dog_data = get_data(dog_path)
    # cat_data = get_data(cat_path)
    os.chdir("./dogs")
    split_data(dog_data, 0.9)
