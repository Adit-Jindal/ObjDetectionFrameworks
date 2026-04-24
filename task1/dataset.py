import os
import pandas as pd
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
from config import *


def map_class(label):
    label = str(label).lower()
    if "human" in label:
        return 1
    else:
        return 0



def convert_bbox(x, y, w, h, img_w, img_h):
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    w = w / img_w
    h = h / img_h
    return xc, yc, w, h



def process_video(video_id, video_path, csv_path, split="train"):
    df = pd.read_csv(csv_path, header=None)

    images_root = video_path

    img_out_dir = IMG_TRAIN if split == "train" else IMG_TEST if split == "test" else IMG_VAL
    lbl_out_dir = LBL_TRAIN if split == "train" else LBL_TEST if split == "test" else LBL_VAL

    for _, row in df.iterrows():

        frame_id = int(row[0])
        frame_str = str(frame_id).zfill(10)

        img_name = f"{video_id}_{frame_str}.jpg"
        img_path = os.path.join(images_root, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        out_img_path = os.path.join(img_out_dir, img_name)
        out_lbl_path = os.path.join(lbl_out_dir, img_name.replace(".jpg", ".txt"))

        if not os.path.exists(out_img_path):
            cv2.imwrite(out_img_path, img)

        x, y, bw, bh = row[2], row[3], row[4], row[5]
        cls = int(row[7])

        xc = (x + bw / 2) / w
        yc = (y + bh / 2) / h
        bw = bw / w
        bh = bh / h

        cls = 1 if cls == 1 else 0

        with open(out_lbl_path, "a") as f:
            f.write(f"{cls} {xc} {yc} {bw} {bh}\n")



def prepare_dataset(root="TrainReal", mode="train"):
    ann_dir = os.path.join(root, "annotations")
    img_dir = os.path.join(root, "images")

    dirs = [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL] if mode == "train" else [IMG_TEST, LBL_TEST]

    for p in dirs:
        if os.path.exists(p):
            shutil.rmtree(p)
        os.makedirs(p)

    videos = os.listdir(img_dir)

    for v in videos:
        video_path = os.path.join(img_dir, v)
        csv_path = os.path.join(ann_dir, f"{v}.csv")

        if not os.path.exists(csv_path):
            continue

        # simple split (you can improve later)
        split = "train" if hash(v) % 5 != 0 else "val"

        process_video(v, video_path, csv_path, split if mode == "train" else mode)

    print("Dataset conversion complete → YOLO format ready.")
