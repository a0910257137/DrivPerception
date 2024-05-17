import numpy as np
import sys
import argparse
import cv2
import os
import json
import tensorflow as tf
import math
from pprint import pprint
from tqdm import tqdm
from box import Box
import random
from pathlib import Path
import copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_2d(box):
    tl = np.asarray([box['y1'], box['x1']])
    br = np.asarray([box['y2'], box['x2']])
    tl = np.expand_dims(tl, axis=-1)
    br = np.expand_dims(br, axis=-1)
    obj_kp = np.concatenate([tl, br], axis=-1)
    obj_kp = np.transpose(obj_kp, [1, 0])
    return obj_kp


def build_2d_obj(obj, obj_cates, img_info):
    obj_kp = get_2d(obj['box2d'])
    obj_name = obj['category'].split(' ')
    cat_key = str()
    for i, l in enumerate(obj_name):
        if i == 0:
            cat_key += l
        else:
            cat_key += '_' + l
    cat_lb = obj_cates[cat_key]

    cat_lb = np.expand_dims(np.asarray([cat_lb, cat_lb]), axis=-1)
    obj_kp = np.concatenate([obj_kp, cat_lb], axis=-1)
    bool_mask = np.isinf(obj_kp).astype(np.float32)

    obj_kp = np.where(obj_kp >= 0., obj_kp, 0.)
    obj_kp[:, 0] = np.where(obj_kp[:, 0] < img_info['height'], obj_kp[:, 0],
                            img_info['height'] - 1)
    obj_kp[:, 1] = np.where(obj_kp[:, 1] < img_info['width'], obj_kp[:, 1],
                            img_info['width'] - 1)
    obj_kp = np.where(bool_mask, np.inf, obj_kp)
    return obj_kp


def make_dir(path):
    if not os.path.exists(path):
        os.umask(0)
        os.makedirs(path, mode=0o755)


def letterbox(combination,
              new_shape=(640, 640),
              color=(114, 114, 114),
              auto=True,
              scaleFill=False,
              scaleup=True):
    """Resize the input image and automatically padding to suitable shape :https://zhuanlan.zhihu.com/p/172121380"""
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    img, gray, line = combination
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
        1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[
            0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        gray = cv2.resize(gray, new_unpad, interpolation=cv2.INTER_LINEAR)
        line = cv2.resize(line, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img,
                             top,
                             bottom,
                             left,
                             right,
                             cv2.BORDER_CONSTANT,
                             value=color)  # add border
    gray = cv2.copyMakeBorder(gray,
                              top,
                              bottom,
                              left,
                              right,
                              cv2.BORDER_CONSTANT,
                              value=0)  # add border
    line = cv2.copyMakeBorder(line,
                              top,
                              bottom,
                              left,
                              right,
                              cv2.BORDER_CONSTANT,
                              value=0)  # add border
    # print(img.shape)

    combination = (img, gray, line)
    return combination, ratio, (dw, dh)


def complement(annos, max_num):
    n, c = annos.shape
    # assign numpy array to avoid > max_num case
    annos = np.asarray([x for x in annos if x.size != 0])

    if len(annos) < max_num:
        complement = max_num - len(annos)
        # number of keypoints
        # this 2 is coors; hard code term
        complement = np.empty([complement, c])
        complement.fill(np.inf)
        complement = complement.astype(np.float32)
        if len(annos) == 0:
            annos = complement
        else:
            annos = np.concatenate([annos, complement])
    else:
        annos = annos[:max_num, ...]

    return annos


def get_coors(img_root,
              img_size,
              anno_path,
              min_num,
              max_obj=None,
              obj_classes=None,
              train_ratio=0.8):

    def is_img_valid(img_path, mode):
        img_info = {}
        if not os.path.exists(img_path):
            print('%s not exist, bypass' % img_path)
            return None, img_info
        img = cv2.imread(img_path, mode)
        if img is None:
            print('Can not read %s, bypass' % img_path)
            return None, img_info
        img_info['height'] = img.shape[0]
        img_info['width'] = img.shape[1]
        if len(img.shape) > 2:
            img_info['channel'] = img.shape[2]
        return img, img_info

    def load_json(anno_path):
        with open(anno_path) as f:
            return json.loads(f.read())

    anno = load_json(anno_path)
    discard_imgs = Box({'invalid': 0, 'less_than': 0})
    obj_counts = Box({'total_2d': 0})
    obj_cates = {k: i for i, k in enumerate(obj_classes)}
    num_frames = len(anno['frame_list'])
    num_train_files = math.ceil(num_frames * train_ratio)
    num_test_files = num_frames - num_train_files
    save_root = os.path.abspath(os.path.join(img_root, os.pardir,
                                             'tf_records'))
    frame_count = 0
    num_seg_class = 2
    for frame in tqdm(anno['frame_list'][1000:]):
        num_train_files -= 1
        frame_kps = []
        dataset = frame['dataset']
        img_name = frame['name']
        img_path = os.path.join(img_root, img_name)
        mask_path = os.path.join(img_root, "../annos/da_seg_annotations/imgs",
                                 img_name.replace(".jpg", ".png"))
        lane_path = os.path.join(img_root, "../annos/ll_seg_annotations/imgs",
                                 img_name.replace(".jpg", ".png"))

        img, img_info = is_img_valid(img_path,
                                     mode=cv2.IMREAD_COLOR
                                     | cv2.IMREAD_IGNORE_ORIENTATION)
        if num_seg_class == 3:
            mask_img, _ = is_img_valid(mask_path)
        else:
            mask_img, _ = is_img_valid(mask_path, mode=0)
        lane_img, _ = is_img_valid(lane_path, mode=0)
        if not img_info or len(frame['labels']) == 0 or img is None:
            discard_imgs.invalid += 1
            continue
        img_size = img_size[::-1]
        resized_shape = np.max(img_size)
        h0, w0 = img_info['height'], img_info['width']
        r = resized_shape / max(h0, w0)
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                             interpolation=interp)
            mask_img = cv2.resize(mask_img, (int(w0 * r), int(h0 * r)),
                                  interpolation=interp)
            lane_img = cv2.resize(lane_img, (int(w0 * r), int(h0 * r)),
                                  interpolation=interp)
        # h, w = img.shape[:2]
        (img, mask_img, lane_img), ratio, pad = letterbox(
            (img, mask_img, lane_img),
            int(resized_shape),
            auto=True,
            scaleup=True)
        scale_factor = np.array([r, r])
        for obj in frame['labels']:
            bbox = build_2d_obj(obj, obj_cates, img_info)
            frame_kps.append(bbox)
            obj_counts.total_2d += 1
            if min_num > len(frame_kps):
                discard_imgs.less_than += 1
                continue
        frame_kps = np.asarray(frame_kps)
        det_label, cates = frame_kps[:, :, :2], frame_kps[:, :, -1]
        det_label = np.einsum('n c d, d -> n c d', det_label, np.array([r, r]))
        det_label = det_label[..., ::-1].astype(np.int32)
        n, _, _ = det_label.shape
        det_label = det_label.reshape([n, 4])
        det_label = np.concatenate([np.zeros(shape=[n, 1]), det_label],
                                   axis=-1)
        if len(det_label) > 0:
            labels = det_label.copy()
            labels[:, 1] = ratio[0] * det_label[:, 1] + pad[0]  # pad width
            labels[:, 2] = ratio[1] * det_label[:, 2] + pad[1]  # pad height
            labels[:, 3] = ratio[0] * det_label[:, 3] + pad[0]
            labels[:, 4] = ratio[1] * det_label[:, 4] + pad[1]
        labels = np.concatenate([cates[:, :-1], labels], axis=-1)
        labels = complement(np.asarray(labels, dtype=np.float32), max_obj)
        # raw image
        imgT = img[..., ::-1]
        # convert to tf records bytes
        if num_seg_class == 3:
            _, seg0 = cv2.threshold(mask_img[:, :, 0], 128, 255,
                                    cv2.THRESH_BINARY)
            _, seg1 = cv2.threshold(mask_img[:, :, 1], 1, 255,
                                    cv2.THRESH_BINARY)
            _, seg2 = cv2.threshold(mask_img[:, :, 2], 1, 255,
                                    cv2.THRESH_BINARY)
        else:
            _, seg1 = cv2.threshold(mask_img, 1, 255, cv2.THRESH_BINARY)
            _, seg2 = cv2.threshold(mask_img, 1, 255, cv2.THRESH_BINARY_INV)
        _, lane1 = cv2.threshold(lane_img, 1, 255, cv2.THRESH_BINARY)
        _, lane2 = cv2.threshold(lane_img, 1, 255, cv2.THRESH_BINARY_INV)

        lane1 = lane1 / 255.
        lane2 = lane2 / 255.
        seg1 = seg1 / 255.
        seg2 = seg2 / 255.
        if num_seg_class == 3:
            seg0 = seg0 / 255.
            mask_img = np.stack((seg0, seg1, seg2), axis=-1)
        else:
            mask_img = np.stack((seg2, seg1), axis=-1)
        mask_img = mask_img.astype(np.float32)
        lane_img = np.stack((lane2, lane1), axis=-1)
        lane_img = lane_img.astype(np.float32)
        mask_img = mask_img.tobytes()
        lane_img = lane_img.tobytes()
        imgT = imgT.tobytes()
        labels = labels.tobytes()

        scale_factor = np.asarray(scale_factor, dtype=np.float32).tobytes()
        pad = np.asarray(pad, dtype=np.float32).tobytes()
        if img_path.split('/')[-1].split('.')[-1] == 'png':
            filename = img_path.split('/')[-1].replace('png', 'tfrecords')
        else:
            filename = img_path.split('/')[-1].replace('jpg', 'tfrecords')
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'origin_height': _int64_feature(img_info['height']),
                'origin_width': _int64_feature(img_info['width']),
                'b_images': _bytes_feature(imgT),
                'b_masks': _bytes_feature(mask_img),
                'b_lanes': _bytes_feature(lane_img),
                'b_labels': _bytes_feature(labels),
                'pad': _bytes_feature(pad),
                'scale_factor': _bytes_feature(scale_factor)
            }))
        if num_train_files > 0:
            save_dir = os.path.join(save_root, 'train')
            make_dir(save_dir)
            writer = tf.io.TFRecordWriter(os.path.join(save_dir, filename))
            writer.write(example.SerializeToString())
        else:
            save_dir = os.path.join(save_root, 'test')
            make_dir(save_dir)
            writer = tf.io.TFRecordWriter(os.path.join(save_dir, filename))
            writer.write(example.SerializeToString())
        writer.close()
        frame_count += 1
    output = {
        'total_2d': obj_counts.total_2d,
        'total_frames': num_frames,
        'train_frames': num_train_files,
        'test_frames': num_test_files,
        "discard_invalid_imgs": discard_imgs.invalid,
        "discard_less_than": discard_imgs.less_than
    }
    return output


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_root', type=str)
    parser.add_argument('--anno_file_names', default=None, nargs='+')
    parser.add_argument('--img_root', type=str)
    parser.add_argument('--obj_cate_file', type=str)
    parser.add_argument('--img_size', default=(640, 640), type=tuple)
    parser.add_argument('--max_obj', default=30, type=int)
    parser.add_argument('--min_num', default=1, type=int)
    parser.add_argument('--train_ratio', default=0.8, type=float)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    if args.anno_file_names is None:
        anno_file_names = [
            x for x in os.listdir(args.anno_root) if 'json' in x
        ]
    else:
        anno_file_names = args.anno_file_names

    if len(anno_file_names) == 0:
        print('No annotations, exit')
        sys.exit(0)

    save_root = os.path.abspath(os.path.join(args.anno_root, os.pardir))
    total_discards = Box({
        'invalid': 0,
        'less_than': 0,
    })
    # read object categories
    if args.obj_cate_file:
        with open(args.obj_cate_file) as f:
            obj_cates = f.readlines()
            obj_cates = [x.strip() for x in obj_cates]
    else:
        obj_cates = None
    for anno_file_name in anno_file_names:
        print('Process %s' % anno_file_name)
        anno_path = os.path.join(args.anno_root, anno_file_name)

        output = get_coors(args.img_root,
                           args.img_size,
                           anno_path,
                           args.min_num,
                           max_obj=args.max_obj,
                           obj_classes=obj_cates,
                           train_ratio=args.train_ratio)
        print('generated TF records are saved in %s' % save_root)
        print(
            'Total 2d objs: %i, Total invalid objs: %i, Total less_than objs: %i'
            % (output['total_2d'], output['discard_invalid_imgs'],
               output['discard_less_than']))
