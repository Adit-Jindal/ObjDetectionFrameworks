import argparse
from ultralytics import YOLO
from dataset import prepare_dataset
from eval_utils import run_evaluation
from config import *
import torch



def train(data_dir):
    prepare_dataset(data_dir)

    model = YOLO("yolov8n.pt")

    model.train(
        data="task1/birdsai.yaml",
        epochs=25,
        imgsz=640,
        batch=16,
        project="task1/runs/detect",
        name="yolo_birdsai",
        device="mps" if torch.backends.mps.is_available() else "cpu" if torch.cuda.is_available() else "cpu",
        workers=2,
        cache=False,
    )


def evaluate(data_dir):
    prepare_dataset(data_dir, "test")
    run_evaluation(data_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate"], required=True)
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    if args.mode == "train":
        train(data_dir)
    else:
        evaluate(data_dir)

if __name__ == "__main__":
    main()
