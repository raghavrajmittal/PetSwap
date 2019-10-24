#/usr/bin/bash

wget "https://archive.org/download/CAT_DATASET/CAT_DATASET_01.zip"
wget "https://archive.org/download/CAT_DATASET/CAT_DATASET_02.zip"
wget "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"

unzip -d "cats/" "CAT_DATASET_01.zip"
unzip -d "cats/" "CAT_DATASET_02.zip"
tar -xvf "images.tar" -d "/dogs"

python transform_data.py
