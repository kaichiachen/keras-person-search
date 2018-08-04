import numpy as np
import cv2


def calculate_iou(img_mask, gt_mask):
    gm = gt_mask * 1.0
    img_and = cv2.bitwise_and(img_mask, gm)
    img_or = cv2.bitwise_or(img_mask, gm)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(img_or)
    iou = float(float(j)/float(i))
    return iou


def calculate_overlapping(img_mask, gt_mask):
    gm = gt_mask * 1.0
    img_and = cv2.bitwise_and(img_mask, gm)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(gm)
    overlap = float(float(j)/float(i))
    return overlap


def follow_iou(gt_mask, mask):
    return calculate_iou(mask, gt_mask)
