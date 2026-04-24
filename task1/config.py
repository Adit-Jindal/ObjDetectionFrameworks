import os

OUTPUT_DIR = "task1/birdsai_yolo"

IMG_TRAIN = os.path.join(OUTPUT_DIR, "images/train")
IMG_VAL   = os.path.join(OUTPUT_DIR, "images/val")
LBL_TRAIN = os.path.join(OUTPUT_DIR, "labels/train")
LBL_VAL   = os.path.join(OUTPUT_DIR, "labels/val")
IMG_TEST  = os.path.join(OUTPUT_DIR, "images/test")
LBL_TEST  = os.path.join(OUTPUT_DIR, "labels/test")


VAL_DATA = "Dataset/ValReal"