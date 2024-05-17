import numpy as np
import cv2
from tensorpack.dataflow import *
import tensorflow as tf


# @tf.function
def _flip(b_objs_kps, b_obj_wid, w, channel_names, do_flip):
    if "kp" in channel_names or "keypoint" in channel_names:
        b_objs_x = b_objs_kps[..., 1:2]
        b_objs_x = -b_objs_x + w - 1
        b_objs_y = b_objs_kps[..., :1]
        b_objs_kps = tf.concat([b_objs_y, b_objs_x], axis=-1)
    else:
        tmp_coors = []
        for i, channel_name in enumerate(channel_names):
            kp = b_objs_kps[:, :, i, :]
            if 'tl' in channel_name or 'top_left' in channel_name:
                flip_x = w - kp[:, :, 1] - b_obj_wid
            elif 'br' in channel_name or 'bottom_right' in channel_name:
                flip_x = w - kp[:, :, 1] + b_obj_wid
            elif 'st' in channel_name or 'side_top' in channel_name:
                flip_x = w - kp[:, :, 1] + b_obj_wid
            elif 'sb' in channel_name or 'side_bottom' in channel_name:
                flip_x = w - kp[:, :, 1] + b_obj_wid
            else:
                flip_x = w - kp[:, :, 1]
            flip_kp = tf.concat([kp[:, :, 0, None], flip_x[..., None]],
                                axis=-1)
            tmp_coors.append([flip_kp])
        b_objs_kps = tf.concat(tmp_coors, axis=0)
        b_objs_kps = tf.transpose(b_objs_kps, [1, 2, 0, 3])
    return b_objs_kps


def _flip_human(kps, w):
    dist = np.float32(w / 2 - kps[:, 1])
    x = np.float32(kps[:, 1] + 2 * dist)
    x[np.isnan(x)] = np.inf
    kps[:, 1] = x
    head = np.expand_dims(kps[0], axis=0)
    body = kps[1:]
    body = body.reshape((6, 2, 2))
    body = np.flip(body, axis=1)
    body = body.reshape((12, 2))
    assert len(head.shape) == len(body.shape), f'{head.shape}_{body.shape}'
    kps = np.concatenate([head, body])
    return kps


def _coor_clip(kps, h_thres, w_thres):
    b, n, c = [tf.shape(kps)[i] for i in range(3)]
    kps = tf.reshape(kps, [b, n, 2, 2])
    # clipped the key point(x,y) on the coordinate
    inf_mask = tf.math.is_inf(kps)
    x, y = kps[..., 0], kps[..., 1]
    y = tf.where(y < 0., 0., y)
    x = tf.where(x < 0., 0., x)
    y = tf.expand_dims(tf.where(y < h_thres, y, h_thres), axis=-1)
    x = tf.expand_dims(tf.where(x < w_thres, x, w_thres), axis=-1)
    result = tf.concat([x, y], axis=-1)
    result = tf.where(inf_mask, np.inf, result)
    result = tf.reshape(result, [b, n, -1])
    return result


def debug_kps(img,
              coors,
              origin_img_size,
              coor_resize,
              task,
              name='output.jpg'):
    img = img[..., ::-1]
    # origin_img_size = origin_img_size.numpy()
    if img.shape[0] != origin_img_size[0] or img.shape[1] != origin_img_size[1]:
        img = cv2.resize(img, tuple(origin_img_size[::-1]))
    # img = np.asarray(img)*255.0
    img = np.asarray(img) * 1.0
    coors = coors.numpy()
    resize_factor = (1 / coor_resize)
    for obj_kps in coors:
        if np.all(np.isinf(obj_kps)) or np.all(np.isinf(obj_kps)):
            continue
        if 'rel' in str(task):
            cates = obj_kps[..., -1:]
            obj_kps = obj_kps[..., :4].reshape([-1, 2])
        obj_kps = np.einsum('n d, d ->n d', obj_kps, resize_factor)
        obj_kps = (obj_kps + 0.5).astype(int)
        coor_n, yx = obj_kps.shape
        if coor_n == 2:
            tl, br = obj_kps
            img = cv2.rectangle(img, tuple(tl[::-1]), tuple(br[::-1]),
                                (255, 0, 0), 3)
        elif coor_n == 3:
            center, tl, br = obj_kps
            img = cv2.circle(img, tuple(center[::-1]), 3, (0, 255, 0), -1)
        elif coor_n == 4:
            obj_tl, obj_br = obj_kps[0], obj_kps[1]
            sub_tl, sub_br = obj_kps[2], obj_kps[3]
            img = cv2.rectangle(img, tuple(obj_tl[::-1]), tuple(obj_br[::-1]),
                                (255, 0, 0), 3)
            img = cv2.rectangle(img, tuple(sub_tl[::-1]), tuple(sub_br[::-1]),
                                (0, 255, 0), 3)
            # img = cv2.circle(img, tuple(center[::-1]), 3, (0, 255, 0), -1)
    cv2.imwrite(name, img)
