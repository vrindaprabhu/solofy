from glob import glob
import torch
import cv2
import numpy as np
from torch.autograd import Variable
from lib.yolov3.darknet import Darknet
from lib.yolov3.util import (
    process_result,
    load_images,
    resize_image,
    cv_image2tensor,
    transform_result,
)
import pickle as pkl
import argparse
import math
import random
import os.path as osp
import os
import sys
import ray
from datetime import datetime
from config.solofy import *


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def create_batches(imgs, batch_size):
    num_batches = math.ceil(len(imgs) // batch_size)
    batches = [
        imgs[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)
    ]

    return batches


def draw_bbox(imgs, bbox, colors, classes, count):
    img = imgs[int(bbox[0])]
    label = classes[int(bbox[-1])]
    p1 = tuple(bbox[1:3].int())
    p2 = tuple(bbox[3:5].int())
    bboxes = []
    for i in p1:
        bboxes.append(i.item())
    for i in p2:
        bboxes.append(i.item())

    if label == "person":
        color = random.choice(colors)
        cv2.rectangle(img, p1, p2, color, 2)

        text_size = cv2.getTextSize(
            f"{label}-{count}", cv2.FONT_HERSHEY_SIMPLEX, 1, 1
        )[0]
        p3 = (p1[0], p1[1] - text_size[1] - 4)
        p4 = (p1[0] + text_size[0] + 4, p1[1])
        cv2.rectangle(img, p3, p4, color, -1)
        cv2.putText(
            img,
            f"{label}-{count}",
            p1,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            [225, 255, 255],
            1,
        )

        return bboxes


def detect_image(model):

    print("-- YOLO: Loading input image(s) now... --")
    input_size = [int(model.net_info["height"]), int(model.net_info["width"])]
    batch_size = int(model.net_info["batch"])

    imlist, imgs = load_images(INIMAGE)

    img_batches = create_batches(imgs, batch_size)

    # load colors and classes
    colors = pkl.load(
        open(
            "./lib/yolov3/pallete",
            "rb",
        )
    )
    classes = load_classes("./lib/yolov3/data/coco.names")

    start_time = datetime.now()
    print("-- Detecting objects --")
    for img_batch in img_batches:
        img_tensors = [cv_image2tensor(img, input_size) for img in img_batch]
        img_tensors = torch.stack(img_tensors)
        img_tensors = Variable(img_tensors)
        if CUDA:
            img_tensors = img_tensors.cuda()
        detections = model(img_tensors, CUDA).to(GPU_DEVICE)

        detections = process_result(detections, OBJ_THRESH, NMS_THRESH)

        if len(detections) == 0:
            continue

        detections = transform_result(detections, img_batch, input_size)
        bboxes = {}
        for count, detection in enumerate(detections):
            bbox = draw_bbox(img_batch, detection, colors, classes, count + 1)
            if bbox:
                bboxes[f"Person-{count+1}"] = bbox

        for img in img_batch:
            image_shape = img.shape
            cv2.imwrite(DETIMAGE, img)

    end_time = datetime.now()
    print("-- Detection finished in %s --" % (end_time - start_time))

    return bboxes, image_shape


@ray.remote
def yolov3_main():

    if CUDA and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    print("-- Loading network --")
    model = Darknet("./lib/yolov3/cfg/yolov3.cfg")
    model.load_weights(YOLO_MODEL_PATH)
    if CUDA:
        model.cuda()

    model.eval()
    print("-- Network loaded --")

    bboxes = detect_image(model)
    return bboxes


# if __name__ == "__main__":
#     input = "/Users/vrindaprabhu/Assignments/01_Solofy/version_v0/2.png"
#     main(input)
