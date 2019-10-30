#/usr/bin/bash

wget "https://archive.org/download/CAT_DATASET/CAT_DATASET_01.zip"
wget "https://archive.org/download/CAT_DATASET/CAT_DATASET_02.zip"
wget "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"

unzip "CAT_DATASET_01.zip" -d "cats/" 
unzip "CAT_DATASET_02.zip" -d "cats/" 
tar -xvf "images.tar" -C "/dogs" 

python transform_data.py
