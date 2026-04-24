import os
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset

class BirdsAIDataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = root
        self.split = split

        self.img_dir = os.path.join(root, "images")
        self.ann_dir = os.path.join(root, "annotations")

        self.samples = []
        self._build_index()

    def _build_index(self):
        videos = os.listdir(self.img_dir)

        for v in videos:
            csv_path = os.path.join(self.ann_dir, f"{v}.csv")
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path, header=None)

            for _, row in df.iterrows():
                frame_id = int(row[0])
                frame_str = str(frame_id).zfill(10)

                img_name = f"{v}_{frame_str}.jpg"
                img_path = os.path.join(self.img_dir, v, img_name)

                if not os.path.exists(img_path):
                    continue

                self.samples.append((img_path, row))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, row = self.samples[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        x, y, bw, bh = row[2], row[3], row[4], row[5]

        x1 = x
        y1 = y
        x2 = x + bw
        y2 = y + bh

        label = 1 if int(row[7]) == 1 else 0

        target = {
            "boxes": torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32),
            "labels": torch.tensor([label + 1])  # +1 for background
        }

        img = torch.tensor(img / 255., dtype=torch.float32).permute(2, 0, 1)

        return img, target