import tensorflow as tf
import os
import time
import numpy as np
import cv2
from pprint import pprint
from utils.io import load_BFM
import random


class EmbeddingMap(tf.keras.callbacks.Callback):

    def __init__(self,
                 config,
                 writers,
                 train_datasets,
                 test_datasets,
                 update_freq,
                 feed_inputs_display=None):
        super(EmbeddingMap, self).__init__()

        self.train_seen = 0
        self.eval_seen = 0
        self.update_freq = update_freq
        self.feed_inputs_display = feed_inputs_display
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.config = config
        self.data_cfg = self.config.data_reader
        self.max_obj_num = self.data_cfg.max_obj_num
        self.resize_size = tf.cast(self.data_cfg.resize_size, tf.float32)
        self.cates = self._get_cates(self.data_cfg.tasks)
        self.writers = writers
        self.keys = ["center"]
        self.batch_size = config.models.batch_size
        #NOTE: Sperate by different task for 3dmm
        # tdmm = self.config.models['3dmm']
        # self.n_s, self.n_Rt = tdmm["n_s"], tdmm["n_Rt"]
        # self.n_shp, self.n_exp = tdmm["n_shp"], tdmm["n_exp"]
        # self.head_model = load_BFM(tdmm['model_path'])
        # kpt_ind = self.head_model['kpt_ind']
        # X_ind_all = np.stack([kpt_ind * 3, kpt_ind * 3 + 1, kpt_ind * 3 + 2])
        # X_ind_all = tf.concat([
        #     X_ind_all[:, :17], X_ind_all[:, 17:27], X_ind_all[:, 36:48],
        #     X_ind_all[:, 27:36], X_ind_all[:, 48:68]
        # ],
        #                       axis=-1)
        # valid_ind = tf.reshape(tf.transpose(X_ind_all), (-1))
        # self.shapeMU = tf.gather(
        #     tf.cast(self.head_model['shapeMU'], tf.float32), valid_ind)
        # self.shapePC = tf.gather(
        #     tf.cast(self.head_model['shapePC'][:, :50], tf.float32), valid_ind)
        # self.expPC = tf.gather(
        #     tf.cast(self.head_model['expPC'][:, :29], tf.float32), valid_ind)
        # pms = np.load(tdmm['pms_path'])
        # self.train_mean_std = tf.cast(pms[:2, ], tf.float32)
        # self.test_mean_std = tf.cast(pms[2:, ], tf.float32)

    def _get_cates(self, tasks):

        def read(path):
            with open(path) as f:
                return [x.strip() for x in f.readlines()]

        return {
            infos['preprocess']: read(infos["category_path"])
            for infos in tasks
        }

    def on_train_begin(self, logs=None):
        self.train_iter_ds = iter(self.train_datasets)

    def on_test_begin(self, logs=None):
        self.test_iter_ds = iter(self.test_datasets)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.train_seen += 1
        if batch % self.update_freq == 0:
            batch_images, batch_labels = next(self.train_iter_ds)
            fmaps = self.model.model(batch_images, training=False)
            for lb_name in batch_labels:
                if 'heat_map' in lb_name:
                    pred_hms = tf.cast(fmaps[lb_name], tf.float32)
                    gt_hms = tf.cast(batch_labels[lb_name], tf.float32)
                    self._summary_hms(self.writers['train'], batch_images,
                                      gt_hms, pred_hms, lb_name, self.cates,
                                      self.train_seen)
                # elif 'params' == lb_name:
                #     pred_params = tf.cast(fmaps['obj_param_map'], tf.float32)
                #     gt_params = tf.cast(batch_labels[lb_name], tf.float32)
                #     self._summary_3dmm(self.writers['train'], batch_images,
                #                        batch_labels['b_coords'], gt_params,
                #                        pred_params, self.train_seen)

    def on_test_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.eval_seen += 1
        if batch % self.update_freq == 0:
            batch_images, batch_labels = next(self.test_iter_ds)
            fmaps = self.model.model(batch_images, training=False)
            for lb_name in batch_labels:
                if 'heat_map' in lb_name:
                    pred_hms = tf.cast(fmaps[lb_name], tf.float32)
                    gt_hms = tf.cast(batch_labels[lb_name], tf.float32)
                    self._summary_hms(self.writers['validation'], batch_images,
                                      gt_hms, pred_hms, lb_name, self.cates,
                                      self.eval_seen)
                # elif 'params' == lb_name:
                #     pred_params = tf.cast(fmaps['obj_param_map'], tf.float32)
                #     gt_params = tf.cast(batch_labels[lb_name], tf.float32)
                #     self._summary_3dmm(self.writers['train'], batch_images,
                #                        batch_labels['b_coords'], gt_params,
                #                        pred_params, self.eval_seen)

    def _summary_hms(self, writer, batch_images, gt_hms, pred_hms, lb_name,
                     task_cates, saved_step):
        """
            Args:
                batch_images: g.t. images
                gt_hms:  g.t. labeld hms
                pred_hms: pred. hms
                task_cates:   {task_name1:[], task_name2:[]}
                saved_step: int saved step
        """
        with writer.as_default():
            tf.summary.image(name='1_image',
                             data=batch_images,
                             step=saved_step,
                             max_outputs=2)
            # list type for task_cates
            for task_name in task_cates:
                if 'rel' in task_name and 'rel' in lb_name:
                    cates = task_cates[task_name]
                    for j, cate in enumerate(cates):
                        pred_hm = tf.expand_dims(pred_hms[..., j], axis=-1)
                        gt_hm = tf.expand_dims(gt_hms[..., j], axis=-1)
                        tf.summary.image(name='heatmap/{}'.format(cate.lower()),
                                         data=pred_hm,
                                         step=saved_step,
                                         max_outputs=2)
                        tf.summary.image(name='heatmap/{}'.format(cate.lower()),
                                         data=gt_hm,
                                         step=saved_step,
                                         max_outputs=2)
                elif 'obj' in task_name or 'tdmm' in task_name:
                    cates = task_cates[task_name]
                    for j, key in enumerate(self.keys):
                        pred_hm = tf.expand_dims(pred_hms[..., j], axis=-1)
                        gt_hm = tf.expand_dims(gt_hms[..., j], axis=-1)
                        tf.summary.image(name='2_gt/{}'.format(key),
                                         data=gt_hm,
                                         step=saved_step,
                                         max_outputs=2)
                        tf.summary.image(name='3_pred/{}'.format(key),
                                         data=pred_hm,
                                         step=saved_step,
                                         max_outputs=2)
        writer.flush()
