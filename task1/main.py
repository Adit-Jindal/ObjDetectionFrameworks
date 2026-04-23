import argparse
from ultralytics import YOLO
from dataset import prepare_dataset
from eval_utils import evaluate_model

def train(data_dir):
    prepare_dataset(data_dir)

    model = YOLO("yolov8n.pt")

    model.train(
        data="task1/birdsai.yaml",
        epochs=25,
        imgsz=640,
        batch=16,
        name="yolo_birdsai",
        device="mps",
    )

def evaluate(data_dir):
    model = YOLO("runs/detect/yolo_birdsai/weights/best.pt")

    metrics = model.val(data="task1/birdsai.yaml")
    print("Overall mAP@0.5:", metrics.box.map50)

    evaluate_model(model, data_dir)

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
