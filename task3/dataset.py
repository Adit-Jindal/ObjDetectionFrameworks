import os
import cv2
import torch
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset

class BirdsAIDataset(Dataset):
    def __init__(self, root, processor):
        self.root = root
        self.processor = processor

        self.img_dir = os.path.join(root, "images")
        self.ann_dir = os.path.join(root, "annotations")

        self.image_data = defaultdict(list)
        self.samples = []
        self._build()

    def _build(self):
        for v in os.listdir(self.img_dir):
            csv_path = os.path.join(self.ann_dir, f"{v}.csv")
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path, header=None)

            for _, row in df.iterrows():
                frame = str(int(row[0])).zfill(10)
                img_name = f"{v}_{frame}.jpg"
                img_path = os.path.join(self.img_dir, v, img_name)

                if os.path.exists(img_path):
                    self.image_data[img_path].append(row)

        self.samples = list(self.image_data.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        rows = self.image_data[path]

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        areas = []

        for r in rows:
            x, y, w, h = r[2], r[3], r[4], r[5]

            boxes.append([x, y, w, h])   # COCO format
            labels.append(1 if int(r[7]) == 1 else 0)
            areas.append(w * h)

        # handle empty
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [0]
            areas = [1]

        annotations = {
            "image_id": idx,
            "annotations": [
                {
                    "bbox": box,
                    "category_id": label,
                    "area": area,
                    "iscrowd": 0
                }
                for box, label, area in zip(boxes, labels, areas)
            ]
        }

        encoding = self.processor(
            images=image,
            annotations=annotations,
            return_tensors="pt"
        )

        # IMPORTANT: don't squeeze incorrectly
        pixel_values = encoding["pixel_values"][0]
        target = encoding["labels"][0]

        return pixel_values, target


def collate_fn(batch, processor):
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    encoding = processor.pad(
        pixel_values,
        return_tensors="pt"
    )

    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels
    }