import tensorflow as tf
import numpy as np
import cv2
import os
import time
from .core import *
from .utils.augmentations import letterbox_for_img
from pprint import pprint


class DrivePredictor:

    def __init__(self, config=None):
        self.config = config
        self.pred_cfg = self.config['predictor']
        os.environ['CUDA_VISIBLE_DEVICES'] = self.pred_cfg['visible_gpu']
        self.gpu_setting(self.pred_cfg["gpu_fraction"])
        self.model_dir = self.pred_cfg['pb_path']
        self.top_k_n = self.pred_cfg['top_k_n']
        self.img_input_size = self.pred_cfg['img_input_size']
        self.is_plot = self.pred_cfg['is_plot']
        self.h, self.w = self.img_input_size
        self.nms_iou_thres = self.pred_cfg['nms_iou_thres']
        self.model_format = self.pred_cfg['model_format']
        self.kp_thres = self.pred_cfg['kp_thres']
        self.n_objs = self.pred_cfg['n_objs']
        self.nc = self.pred_cfg['nc']
        self.predictor_mode = self.pred_cfg['predictor_mode']
        self._model = tf.keras.models.load_model(self.model_dir)
        self._post_model = DrivePostModel(self._model, self.nc, self.n_objs,
                                          self.top_k_n, self.kp_thres,
                                          self.nms_iou_thres,
                                          self.img_input_size, self.is_plot)

    def pred(self, imgs):
        temp_imgs, temp_ratio, temp_pad = [], [], []
        for img in imgs:
            img, ratio, pad = letterbox_for_img(img,
                                                new_shape=self.w,
                                                auto=True)
            img = img / 255.
            temp_imgs.append(img[..., ::-1])
            temp_ratio.append(ratio)
            temp_pad.append(pad)
        imgs = tf.cast(np.asarray(temp_imgs), tf.float32)
        ratios = tf.cast(np.asarray(temp_ratio), tf.float32)
        paddings = tf.cast(np.asarray(temp_pad), tf.float32)
        rets = self._post_model([imgs, ratios, paddings], training=False)
        return rets

    def gpu_setting(self, fraction):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        gpu_config = tf.compat.v1.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        gpu_config.gpu_options.per_process_gpu_memory_fraction = fraction
        tf.compat.v1.keras.backend.set_session(
            tf.compat.v1.Session(config=gpu_config))
        for i in range(len(gpus)):
            tf.config.experimental.set_memory_growth(gpus[i], True)
