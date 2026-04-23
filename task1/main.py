import argparse
from ultralytics import YOLO
from dataset import prepare_dataset
from eval_utils import evaluate_model
from config import *
import pandas as pd

LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.csv")
ENABLE_LOGGING = True   


def log_metrics(trainer):
    if not ENABLE_LOGGING:
        return

    metrics = trainer.metrics
    epoch = trainer.epoch

    row = {
        "epoch": epoch,
        "train_loss": float(trainer.loss.item()) if trainer.loss is not None else None,
        "val_box_loss": metrics.get("val/box_loss"),
        "val_cls_loss": metrics.get("val/cls_loss"),
        "val_dfl_loss": metrics.get("val/dfl_loss"),
        "precision": metrics.get("metrics/precision(B)"),
        "recall": metrics.get("metrics/recall(B)"),
        "mAP50": metrics.get("metrics/mAP50(B)"),
        "mAP50-95": metrics.get("metrics/mAP50-95(B)")
    }

    df = pd.DataFrame([row])

    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)


def train(data_dir):
    prepare_dataset(data_dir)

    model = YOLO("yolov8n.pt")

    if ENABLE_LOGGING:
        model.add_callback("on_fit_epoch_end", log_metrics)

    model.train(
        data="task1/birdsai.yaml",
        epochs=25,
        imgsz=640,
        batch=16,
        name="yolo_birdsai",
        device="mps",
        workers=2,
        cache=False,
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
