import os
import tensorflow as tf
import multiprocessing
import numpy as np
import cv2
from .general_task import GeneralTasks
from box import Box
from pprint import pprint
from glob import glob

threads = multiprocessing.cpu_count()


class GeneralDataset:

    def __init__(self, config, mirrored_strategy):

        def read_cates(category_path):
            with open(category_path) as f:
                return [x.strip() for x in f.readlines()]

        self.config = config
        self.batch_size = config.batch_size * mirrored_strategy.num_replicas_in_sync
        self.tasks = config.tasks
        self.epochs = config.epochs
        for task in config.tasks:
            task['cates'] = read_cates(task['category_path'])
        self.config = Box(self.config)
        self.gener_task = GeneralTasks(self.config, self.batch_size)

    def _dataset(self, is_train):
        datasets = []
        for task in self.config.tasks:
            if is_train:
                filenames = glob(os.path.join(task.train_folder,
                                              '*.tfrecords'))
                num_files = len(filenames)
                ds = tf.data.TFRecordDataset(filenames,
                                             num_parallel_reads=threads)
            else:
                filenames = glob(os.path.join(task.test_folder, '*.tfrecords'))
                num_files = len(filenames)
                ds = tf.data.TFRecordDataset(filenames,
                                             num_parallel_reads=threads)
            datasets.append(ds)
        datasets = tf.data.TFRecordDataset.zip(tuple(datasets))
        if self.config.shuffle:
            datasets = datasets.shuffle(buffer_size=10000)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        datasets = datasets.with_options(options)
        datasets = datasets.batch(self.batch_size, drop_remainder=True)
        # (384, 640)
        # for ds in datasets:
        #     b_img, targets = self.gener_task.build_maps(ds)
        #     b_img = b_img.numpy() * 255
        #     b_xywh = targets['b_xywh'].numpy()
        #     b_paddings = targets['b_paddings'].numpy()
        #     for img, xywh, paddings in zip(b_img, b_xywh, b_paddings):
        #         mask = np.all(np.isfinite(xywh), axis=-1)
        #         xywhs = xywh[mask]
        #         for nxywh in xywhs:
        #             i, c, nxywh = nxywh[0], nxywh[1], nxywh[2:]
        #             nxywh *= np.array([640, 384, 640, 384], dtype=np.float32)
        #             center_xy, obj_wh = nxywh[:2], nxywh[2:]
        #             tl = (center_xy - obj_wh / 2).astype(np.int32)
        #             br = (center_xy + obj_wh / 2).astype(np.int32)
        #             img = cv2.rectangle(img, tuple(tl), tuple(br), (0, 255, 0),
        #                                 3)
        #         cv2.imwrite("output.jpg", img[..., ::-1])
        datasets = datasets.map(
            lambda *x: self.gener_task.build_maps(x),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets, num_files

    def get_datasets(self):
        training_ds, num_training_ds = self._dataset(True)
        testing_ds, num_testing_ds = self._dataset(False)
        return {
            "train": training_ds,
            "test": testing_ds,
            "training_length": num_training_ds,
            "testing_length": num_testing_ds
        }
