# # Mask R-CNN
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import argparse


def get_masks(img):

    ROOT_DIR = os.path.abspath("mask/")  # Root directory of the project
    sys.path.append(ROOT_DIR)  # To find local version of the library
    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
    MODEL_DIR = os.path.join(
        ROOT_DIR, "logs"
    )  # Directory to save logs and trained model
    COCO_MODEL_PATH = os.path.join(
        ROOT_DIR, "mask_rcnn_coco.h5"
    )  # Local path to trained weights file
    from mrcnn import utils
    import mrcnn.model as modellib
    from mrcnn import visualize
    import coco

    if not os.path.exists(
        COCO_MODEL_PATH
    ):  # Download COCO trained weights from Releases if needed
        utils.download_trained_weights(COCO_MODEL_PATH)
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")

    # in case video needs to be saved
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--images",
        default=os.path.join(ROOT_DIR, "images"),
        help="path of the image directory",
    )
    args = vars(ap.parse_args())
    IMAGE_DIR = args["images"]

    # ## Configurations
    # We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
    # For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.
    class InferenceConfig(coco.CocoConfig):
        # Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    #config.display()

    # ## Create Model and Load Trained Weights
    model = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_DIR, config=config
    )  # Create model object in inference mode.
    model.load_weights(COCO_MODEL_PATH, by_name=True)  # Load weights trained on MS-COCO

    # ## Class Names
    # To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
    # To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
    # dataset = coco.CocoDataset()
    # dataset.load_coco(COCO_DIR, "train")
    # dataset.prepare()
    # print(dataset.class_names)
    # COCO Class names - Instead of downloading, we're including the list of class names below
    # Index of the class in the list is its ID.
    class_names = [
        "BG",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    # print(class_names.index('cat')) # -> 16
    # print(class_names.index('dog')) # -> 17

    # ## Run Object Detection
    # Load a random image from the images folder and visualize results
    file_names = next(os.walk(IMAGE_DIR))[2]
    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    results = model.detect([img], verbose=0)  # Run detection

    # filter out non chosen classes
    chosen_classes = [16, 17]  # class indices for cat, dog
    for r in results:
        index_mask = np.isin(r["class_ids"], chosen_classes)
        r["rois"] = r["rois"][index_mask]
        r["class_ids"] = r["class_ids"][index_mask]
        r["scores"] = r["scores"][index_mask]
        r["masks"] = r["masks"][:, :, index_mask]
    return results
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
