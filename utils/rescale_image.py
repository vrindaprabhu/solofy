import cv2
import numpy as np
from config.solofy import *


def rescale_image(bboxes, original_h, original_w):
    img = cv2.imread(INIMAGE)
    img = cv2.resize(img, RESIZE_TO)

    w = RESIZE_TO[0] / original_w
    h = RESIZE_TO[1] / original_h
    scaler = np.asarray([w, h, w, h])
    scaled_boxes = np.asarray(list(bboxes.values())) * scaler
    scaled_boxes = scaled_boxes.astype(np.uint32)

    for bbox in scaled_boxes:
        resized_detection = cv2.rectangle(
            img,
            pt1=(bbox[0], bbox[1]),
            pt2=(bbox[2], bbox[3]),
            color=(255, 255, 255),
            thickness=2,
        )

    return scaled_boxes, resized_detection
