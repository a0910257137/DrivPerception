from re import X
import numpy as np
import tensorflow as tf
from .utils import _coor_clip


class ObjDet:

    def offer_kps(self, b_objs_kps, h, w):
        b_objs_kps = tf.where(b_objs_kps > 1e8, np.inf, b_objs_kps)
        b_objs_kps = _coor_clip(b_objs_kps, h - 1, w - 1)
        x1, y1 = b_objs_kps[:, :, 0], b_objs_kps[:, :, 1]
        x2, y2 = b_objs_kps[:, :, 2], b_objs_kps[:, :, 3]
        xywh = self.convert((w, h), (x1, x2, y1, y2))
        return xywh

    def convert(self, size, box):
        dw = 1. / (size[0])  # 640
        dh = 1. / (size[1])  # 384
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = (x * dw)[..., None]
        w = (w * dw)[..., None]
        y = (y * dh)[..., None]
        h = (h * dh)[..., None]
        xywh = tf.concat([x, y, w, h], axis=-1)
        xywh = tf.where(tf.math.is_nan(xywh), np.inf, xywh)
        return xywh
