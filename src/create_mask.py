import numpy as np
import cv2
from config.solofy import *


def create_mask_inpaint(mask_bbox, img_shape):
    mask = np.zeros(shape=img_shape, dtype="uint8")
    extra = EXTRA_PIXEL
    for i in mask_bbox:
        mask = cv2.rectangle(
            img=mask,
            pt1=(i[0] - extra, i[1] - extra),
            pt2=(i[2] + extra, i[3] + extra),
            color=(255, 255, 255),
            thickness=-1,
        )
    mask = cv2.resize(mask, RESIZE_TO)
    cv2.imwrite(MASKIMAGE, mask)


def create_mask_bgreplace(mask_bbox):
    bg_image = cv2.imread(BGIMAGE)
    bg_image = bg_image.astype(float)

    fg_image = cv2.imread(INIMAGE)
    fg_image = cv2.resize(fg_image, RESIZE_TO)
    fg_image = fg_image.astype(float)

    fg_mask = cv2.imread(SALIENTIMAGE)
    _, mask = cv2.threshold(fg_mask, 75, 255, cv2.THRESH_BINARY)
    alpha = np.zeros(fg_mask.shape, dtype="uint16")

    mask_bbox = mask_bbox + np.asarray(
        [-EXTRA_FG_PIXEL, -EXTRA_FG_PIXEL, EXTRA_FG_PIXEL, EXTRA_FG_PIXEL]
    )
    mask_bbox[mask_bbox > RESIZE_TO[0]] = RESIZE_TO[0]
    mask_bbox[mask_bbox < 0] = 0
    y1, x1, y2, x2 = mask_bbox

    alpha[x1:x2, y1:y2] = mask[x1:x2, y1:y2]
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
    alpha = alpha.astype(float) / 255

    background = 255 * np.ones_like(fg_image).astype(np.uint8)
    background = background.astype(float)
    foreground = cv2.multiply(alpha, fg_image)
    background = cv2.multiply(1.0 - alpha, bg_image)

    outimage = cv2.add(foreground, background)
    cv2.imwrite(OUTIMAGE, outimage)
