import random
import tensorflow as tf
from .mosaic import Mosaic
import logging


class MultiAugmentation:

    def __init__(self, process_type, batch_size, p, need_num):
        self.process_type = process_type
        self.batch_size = batch_size
        self.p = p
        self.need_num = need_num

    #@tf.function
    def __call__(self, batch_images, batch_boxes):
        # Check batch size
        if batch_images.shape[0] < self.need_num:
            warn_mess = f"Need at least batch size = {self.need_num} for augmentation {self.process_type}, but only got {batch_images.shape[0]}, return normal image"
            logging.warning(warn_mess)
            return batch_images, batch_boxes
        tf_images, tf_boxes = [], []
        for num_idx in tf.range(self.batch_size):
            random_num = random.random()
            if random_num < self.p:
                select_images, select_boxes = self.get_images_boxes(
                    batch_images, batch_boxes, num_idx)
                mosaic = Mosaic(process_type=self.process_type,
                                out_size=batch_images.shape[1:3])
                mosaic_image, mosaic_boxes = mosaic(select_images, select_boxes)
                if mosaic_image is None:
                    tf_images.append(batch_images[num_idx])
                    tf_boxes.append(batch_boxes[num_idx])
                else:
                    tf_images.append(mosaic_image)
                    tf_boxes.append(mosaic_boxes)
            else:
                tf_images.append(batch_images[num_idx])
                tf_boxes.append(batch_boxes[num_idx])
        tf_images = tf.cast(tf_images, tf.uint8)
        tf_boxes = tf.cast(tf_boxes, tf.float32)
        return tf_images, tf_boxes

    #@tf.function
    def get_images_boxes(self, batch_images, batch_boxes, num_idx):
        select_idx = random.sample(range(self.batch_size), self.need_num)
        if num_idx not in select_idx:
            select_idx[0] = num_idx
        select_images = tf.gather(batch_images, select_idx)
        select_boxes = tf.gather(batch_boxes, select_idx)
        return select_images, select_boxes