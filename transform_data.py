# /usr/bin/python
import numpy as np
import os
from shutil import copy
from random import shuffle


def get_data(dataset_dir):
    images_list = []
    for d in os.walk(dataset_dir):
        for file in os.listdir(d[0]):
            if file.endswith(".jpg"):
                images_list.append(os.path.join(os.getcwd(), d[0], file))
    shuffle(images_list)
    return images_list


def split_data(dataset, split_val):
    if not os.path.exists("train"):
        os.mkdir("train")
    if not os.path.exists("test"):
        os.mkdir("test")
    i = 0
    while i < len(dataset):
        if i < int(split_val * len(dataset)):
            copy(dataset[i], "./train")
        else:
            copy(dataset[i], "./test")
        i = i + 1
    os.chdir("..")


if __name__ == "__main__":
    dog_path = "./dogs/Images"
    cat_path = "./cats/"
    dog_data = get_data(dog_path)
    cat_data = get_data(cat_path)
    os.chdir("./dogs")
    split_data(dog_data, 0.95)
    os.chdir("./cats")
    split_data(cat_data, 0.95)
