import numpy as np
import tensorflow as tf
import math


def smooth_BCE(eps=0.1):
    # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def bbox_iou(box1,
             box2,
             x1y1x2y2=True,
             GIoU=False,
             DIoU=False,
             CIoU=False,
             eps=1e-9):

    box1 = tf.transpose(box1, perm=[1, 0])
    box2 = tf.transpose(box2, perm=[1, 0])
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    # Intersection area
    x_clip_vals = tf.clip_by_value(
        (tf.math.minimum(b1_x2, b2_x2) - tf.math.maximum(b1_x1, b2_x1)),
        clip_value_min=0,
        clip_value_max=np.inf)
    y_clip_vals = tf.clip_by_value(
        (tf.math.minimum(b1_y2, b2_y2) - tf.math.maximum(b1_y1, b2_y1)),
        clip_value_min=0,
        clip_value_max=np.inf)
    inter = x_clip_vals * y_clip_vals

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = tf.math.maximum(b1_x2, b2_x2) - tf.math.minimum(
            b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = tf.math.maximum(b1_y2, b2_y2) - tf.math.minimum(
            b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2)**2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2)**
                    2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * tf.math.pow(
                    tf.math.atan(w2 / h2) - tf.math.atan(w1 / h1), 2)
                alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU
