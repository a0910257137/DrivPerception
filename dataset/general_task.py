import tensorflow as tf
import numpy as np
from .utils import *
from pprint import pprint
from .preprocess import OFFER_ANNOS_FACTORY
from .augmentation.augmentation import Augmentation
import cv2


class GeneralTasks:

    def __init__(self, config, batch_size):
        self.config = config
        self.task_configs = config['tasks']
        self.model_name = self.config.model_name
        self.map_height, self.map_width = tf.cast(
            self.config.resize_size, tf.float32) * self.config.img_down_ratio
        self.is_do_filp = self.config.augments.do_flip
        self.img_resize_size = tf.cast(self.config.resize_size, dtype=tf.int32)
        self.max_obj_num = self.config.max_obj_num
        self.batch_size = batch_size
        self.img_channel = 3
        self.features = {
            "origin_height": tf.io.FixedLenFeature([], dtype=tf.int64),
            "origin_width": tf.io.FixedLenFeature([], dtype=tf.int64),
            "b_images": tf.io.FixedLenFeature([], dtype=tf.string),
            "b_masks": tf.io.FixedLenFeature([], dtype=tf.string),
            "b_lanes": tf.io.FixedLenFeature([], dtype=tf.string),
            "pad": tf.io.FixedLenFeature([], dtype=tf.string),
            "b_labels": tf.io.FixedLenFeature([], dtype=tf.string),
            "scale_factor": tf.io.FixedLenFeature([], dtype=tf.string)
        }
        self._multi_aug_funcs = Augmentation(self.config, self.batch_size,
                                             self.img_resize_size)

    def build_maps(self, task_infos):
        targets = {}
        for task_infos, infos in zip(self.task_configs, task_infos):
            task, m_cates = task_infos['preprocess'], len(task_infos['cates'])
            b_origin_sizes, b_labels, b_masks, b_lanes, b_imgs, b_scale_factors, b_paddings = self._parse_TFrecord(
                task, infos)
            b_imgs, b_labels = self._multi_aug_funcs(b_imgs, b_labels,
                                                     task_infos.num_lnmks,
                                                     task)
            offer_kps_func = OFFER_ANNOS_FACTORY[task]().offer_kps
            b_xyxy, b_cates = b_labels[..., 2:], b_labels[..., :2]
            # b_obj_sizes = self._obj_sizes(b_xyxy)
            b_xywh = offer_kps_func(b_xyxy, self.map_height, self.map_width)
            # convert to xywh
            b_idxs = tf.tile(
                tf.range(self.batch_size, dtype=tf.float32)[:, None, None],
                [1, self.max_obj_num, 1])
            b_xywh = tf.concat([b_idxs, b_cates[..., 1:], b_xywh], axis=-1)
            targets['b_xywh'] = b_xywh
            targets['b_masks'] = tf.cast(b_masks, tf.float32)
            targets['b_lanes'] = tf.cast(b_lanes, tf.float32)
            targets['b_paddings'] = b_paddings
        return tf.cast(b_imgs, dtype=tf.float32), targets

    @tf.function
    def _obj_sizes(self, b_objs_kps):
        # B, N, 2, 2
        b_obj_sizes = b_objs_kps[:, :, 1, :] - b_objs_kps[:, :, 0, :]
        b_obj_sizes = tf.where(tf.math.is_nan(b_obj_sizes), np.inf,
                               b_obj_sizes)
        return b_obj_sizes

    def _rounding_offset(self, b_kp_idxs, b_round_kp_idxs):
        return b_kp_idxs - b_round_kp_idxs

    @tf.function
    def _one_hots(self, b_cates, m_cates):
        rel_classes = tf.zeros(shape=(self.batch_size, self.max_obj_num,
                                      m_cates),
                               dtype=tf.dtypes.float32)
        is_finites = ~tf.math.is_inf(b_cates)
        b_index = tf.where(is_finites)
        valid_counts = tf.where(is_finites, 1, 0)
        valid_counts = tf.math.reduce_sum(valid_counts)
        class_idx = tf.gather_nd(b_cates, b_index)
        b_index = tf.cast(b_index, tf.float32)
        b_index = tf.concat([b_index, class_idx[:, None]], axis=-1)
        b_index = tf.cast(b_index, tf.int32)
        one_hot_code = tf.tensor_scatter_nd_update(rel_classes, b_index,
                                                   tf.ones(shape=valid_counts))
        return one_hot_code

    def _parse_TFrecord(self, task, infos):
        if (task == "obj_det"):
            anno_shape = [-1, self.max_obj_num, 6]

        b_labels, b_images, b_origin_sizes = None, None, None
        parse_vals = tf.io.parse_example(infos, self.features)
        b_images = tf.io.decode_raw(parse_vals['b_images'], tf.uint8)
        b_images = tf.reshape(
            b_images, [-1, self.map_height, self.map_width, self.img_channel])
        b_masks = tf.io.decode_raw(parse_vals['b_masks'], tf.float32)
        b_masks = tf.reshape(b_masks, [-1, self.map_height, self.map_width, 2])
        b_lanes = tf.io.decode_raw(parse_vals['b_lanes'], tf.float32)
        b_lanes = tf.reshape(b_lanes, [-1, self.map_height, self.map_width, 2])
        b_labels = tf.io.decode_raw(parse_vals['b_labels'], tf.float32)
        b_labels = tf.reshape(b_labels, anno_shape)
        origin_height = tf.reshape(parse_vals['origin_height'], (-1, 1))
        origin_width = tf.reshape(parse_vals['origin_width'], (-1, 1))
        b_origin_sizes = tf.concat([origin_height, origin_width], axis=-1)
        b_origin_sizes = tf.cast(b_origin_sizes, tf.int32)
        b_scale_factors = tf.io.decode_raw(parse_vals['scale_factor'],
                                           tf.float32)  # ratio wh
        b_paddings = tf.io.decode_raw(parse_vals['pad'], tf.float32)  # wh
        return b_origin_sizes, b_labels, b_masks, b_lanes, b_images, b_scale_factors, b_paddings
