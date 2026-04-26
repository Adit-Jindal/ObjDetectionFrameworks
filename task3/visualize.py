import cv2

def draw_boxes(image, boxes, scores, thresh=0.5):
    for box, score in zip(boxes, scores):
        if score < thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
    return image