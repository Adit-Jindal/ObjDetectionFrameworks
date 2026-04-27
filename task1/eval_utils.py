import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict
import shutil
from config import IMG_TEST, LBL_TEST

# ===================== CONFIG =====================

MODEL_PATH = "task1/runs/detect/yolo_birdsai/weights/best.pt"
PRED_SAVE_DIR = "task1/runs/predictions"

# ===================== MAIN =====================

def run_evaluation(data_dir="Dataset/TestReal"):
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    
    if os.path.exists(PRED_SAVE_DIR):
        shutil.rmtree(PRED_SAVE_DIR)
    os.makedirs(PRED_SAVE_DIR)

    print("Running inference on test set...")
    results = model.predict(
        source=IMG_TEST,
        conf=0.25,
        save=True,
        save_txt=True,
        project=os.path.join(os.getcwd(),PRED_SAVE_DIR),
        name="preds",
        verbose=False,
    )

    save_dir = results[0].save_dir
    print(save_dir)

    pred_labels_root = os.path.join(save_dir, "../labels")
    annotations_dir = os.path.join(data_dir, "annotations")

    print("Computing video scales...")
    video_scales = compute_video_scales(annotations_dir)

    print("Evaluating predictions...")
    stats = evaluate_dataset(LBL_TEST, pred_labels_root, video_scales)

    print("Computing mAP table...")
    table = compute_map(stats)

    print_table(table, save_dir)

    print("Showing qualitative results...")
    show_samples(save_dir, n=5)

# ===================== SCALE =====================

def compute_video_scales(annotations_dir):
    video_scale = {}

    for file in os.listdir(annotations_dir):
        if not file.endswith(".csv"):
            continue

        vid = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(annotations_dir, file), header=None)

        areas = df[4] * df[5]
        avg_area = areas.mean()

        if avg_area < 200:
            scale = "S"
        elif avg_area <= 2000:
            scale = "M"
        else:
            scale = "L"

        video_scale[vid] = scale

    return video_scale

# ===================== IOU =====================

def iou(box1, box2):
    xc1, yc1, w1, h1 = box1
    xc2, yc2, w2, h2 = box2

    x1_min, y1_min = xc1 - w1/2, yc1 - h1/2
    x1_max, y1_max = xc1 + w1/2, yc1 + h1/2

    x2_min, y2_min = xc2 - w2/2, yc2 - h2/2
    x2_max, y2_max = xc2 + w2/2, yc2 + h2/2

    xa = max(x1_min, x2_min)
    ya = max(y1_min, y2_min)
    xb = min(x1_max, x2_max)
    yb = min(y1_max, y2_max)

    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1*h1 + w2*h2 - inter

    return inter / union if union > 0 else 0

# ===================== LOAD LABELS =====================

def load_yolo_labels(path):
    boxes = []
    if not os.path.exists(path):
        return boxes

    with open(path) as f:
        for line in f:
            cls, xc, yc, w, h = map(float, line.strip().split())
            boxes.append((int(cls), xc, yc, w, h))
    return boxes


# ===================== MATCH =====================

def evaluate_image(gt_boxes, pred_boxes, iou_thr=0.5):
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    gt_by_cls = defaultdict(list)
    for i, g in enumerate(gt_boxes):
        gt_by_cls[g[0]].append((i, g[1:]))

    pred_by_cls = defaultdict(list)
    for p in pred_boxes:
        pred_by_cls[p[0]].append(p[1:])

    all_classes = set(gt_by_cls.keys()).union(set(pred_by_cls.keys()))

    for cls in all_classes:
        gts = gt_by_cls[cls]
        preds = pred_by_cls[cls]

        matched_gt = set()
        tp, fp = 0, 0

        for p_box in preds:
            best_iou = 0
            best_idx = -1

            for g_idx, g_box in gts:
                if g_idx in matched_gt:
                    continue
                iou_val = iou(p_box, g_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = g_idx

            if best_iou >= iou_thr:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1

        fn = len(gts) - len(matched_gt)
        
        stats[cls]["tp"] += tp
        stats[cls]["fp"] += fp
        stats[cls]["fn"] += fn

    return stats

# ===================== DATASET EVAL =====================

def evaluate_dataset(gt_root, pred_root, video_scales):
    stats = defaultdict(lambda: {"tp":0,"fp":0,"fn":0})

    if not os.path.exists(gt_root):
        return stats

    for file in os.listdir(gt_root):
        if not file.endswith(".txt"):
            continue

        video = file.split("_")[0]
        scale = video_scales.get(video, "M")

        gt_path = os.path.join(gt_root, file)
        pred_path = os.path.join(pred_root, file) if os.path.exists(pred_root) else ""

        gt_boxes = load_yolo_labels(gt_path)
        pred_boxes = load_yolo_labels(pred_path)

        img_stats = evaluate_image(gt_boxes, pred_boxes)

        for cls, s in img_stats.items():
            key = (scale, cls)
            stats[key]["tp"] += s["tp"]
            stats[key]["fp"] += s["fp"]
            stats[key]["fn"] += s["fn"]

    return stats

# ===================== mAP =====================

def compute_map(stats):
    table = {}

    for (scale, cls), s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        label = ["A", "H"][cls]
        table[f"{scale}{label}"] = precision * recall

    def get_ap(cls_id):
        tp = sum(s["tp"] for (sc, c), s in stats.items() if c == cls_id)
        fp = sum(s["fp"] for (sc, c), s in stats.items() if c == cls_id)
        fn = sum(s["fn"] for (sc, c), s in stats.items() if c == cls_id)
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        return p * r

    table["Animals"] = get_ap(0)
    table["Humans"] = get_ap(1)
    table["Overall"] = (table["Animals"] + table["Humans"]) / 2

    return table

# ===================== PRINT =====================

def print_table(table, save_dir):
    sum_path = os.path.join(save_dir, "../summary.txt")
    with open(sum_path) as f:
        f.write("\n=== mAP Table ===")

        rows = [
            "SA","MA","LA","Animals",
            "SH","MH","LH","Humans","Overall"
        ]

        for r in rows:
            f.write(f"{r}: {table.get(r, 0):.4f}")

# ===================== VISUAL =====================

def show_samples(folder, n=5):
    files = [f for f in os.listdir(folder) if f.endswith(".jpg")][:n]

    for f in files:
        img = cv2.imread(os.path.join(folder, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img)
        plt.title(f)
        plt.axis("off")
        plt.show()

# ===================== ENTRY =====================

if __name__ == "__main__":
    run_evaluation()
