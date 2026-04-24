import cv2
import torch

def draw_predictions(img, boxes, scores, labels, thresh=0.5):
    for box, score in zip(boxes, scores):
        if score < thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    return img