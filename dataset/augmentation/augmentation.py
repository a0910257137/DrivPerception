import tensorflow as tf
from .base import Base
# from .mosaic import Mosaic


class Augmentation(Base):

    def __init__(self, config, batch_size, img_resize_size):
        super(Augmentation, self).__init__()
        self.config = config
        self.max_obj_num = self.config.max_obj_num
        self.augments = self.config.augments
        self.is_do_filp = self.augments.do_flip
        self.img_resize_size = img_resize_size
        self.batch_size = batch_size
        self.img_channel = 3

    def __call__(self, b_imgs, b_labels, num_lnmks, task):
        do_clc, gray_probs, flip_probs = self.random_param()
        if self.is_do_filp:
            filp_imgs = tf.image.flip_left_right(b_imgs)
            tmp_logic = tf.tile(
                flip_probs[:, None, None, None],
                [1, self.img_resize_size[0], self.img_resize_size[1], 3])

            b_imgs = tf.where(tf.math.logical_not(tmp_logic), b_imgs,
                              filp_imgs)

        if len(self.augments.tensorpack_chains) != 0:
            b_imgs, b_labels = tf.py_function(
                self.tensorpack_augs,
                inp=[
                    b_labels, b_imgs, task, flip_probs, self.max_obj_num,
                    self.augments.tensorpack_chains
                ],
                Tout=[tf.uint8, tf.float32])

        if len(self.augments.album_chains.keys()) != 0:
            b_imgs = self.album_augs(self.augments.album_chains, b_imgs)
        if len(self.augments.color_chains) != 0:
            aug_imgs = self.color_aug(b_imgs, self.augments.color_chains)
            tmp_logic = tf.tile(do_clc[:, None, None, None], [
                1, self.img_resize_size[0], self.img_resize_size[1],
                self.img_channel
            ])
            b_imgs = tf.where(tf.math.logical_not(tmp_logic), b_imgs, aug_imgs)
            tmp_logic = tf.tile(gray_probs[:, None, None, None], [
                1, self.img_resize_size[0], self.img_resize_size[1],
                self.img_channel
            ])
            b_gray_imgs = tf.image.grayscale_to_rgb(
                tf.image.rgb_to_grayscale(b_imgs))
            b_imgs = tf.where(tf.math.logical_not(tmp_logic), b_imgs,
                              b_gray_imgs)
        b_imgs = b_imgs / 255
        # for obj det
        if task == "obj_det":
            anno_shape = [self.batch_size, self.max_obj_num, 6]
        b_labels = tf.reshape(b_labels, shape=anno_shape)
        b_imgs = tf.reshape(b_imgs, [
            self.batch_size, self.img_resize_size[0], self.img_resize_size[1],
            self.img_channel
        ])
        return b_imgs, b_labels

    def random_param(self):
        col_thre = 0.5 if len(self.augments.color_chains) else 0.0
        do_col = tf.random.uniform(shape=[self.batch_size],
                                   maxval=1,
                                   dtype=tf.float16) < col_thre

        do_gray = tf.random.uniform(shape=[self.batch_size],
                                    maxval=1,
                                    dtype=tf.float16) < col_thre
        flip_thre = 0.5 if self.augments.do_flip else 0.0
        do_flip = tf.random.uniform(shape=[self.batch_size],
                                    maxval=1,
                                    dtype=tf.float16) < flip_thre
        return do_col, do_gray, do_flip
