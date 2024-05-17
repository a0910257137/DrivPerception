import functools
from typing import Tuple
import tensorflow as tf
import numpy as np
import logging


class Mosaic:
    def __init__(self,
                 process_type: str,
                 out_size: Tuple[int, int] = (680, 680),
                 minimum_mosaic_image_dim: int = 25):
        """
        Args:
        out_size: output mosaic image size. high, width
        n_images: number images to make mosaic
        _minimum_mosaic_image_dim: minimum percentage of out_size dimension should the mosaic be. i.e if out_size is (680,680) and 
        _minimum_mosaic_image_dim is 25 , minimum mosaic sub images dimension will be 25 % of 680.
        """
        self._process_type = process_type
        self._out_size = out_size
        self._minimum_mosaic_image_dim = minimum_mosaic_image_dim
        assert (self._minimum_mosaic_image_dim >
                0), "Minimum Mosaic image dimension should be above 0"

    def __call__(self, images, boxes):
        """Builds mosaic with given images, boxes."""
        boxes_list = self._preprocess_boxes(boxes, tf.shape(boxes)[1])
        if boxes_list is None:
            return None, None
        y, x = self._mosaic_divide_points()
        mosaic_sub_images, mosaic_sub_boxes = self._mosaic(
            images, boxes_list, mosaic_divide_points=(y, x))
        mosaic_image = self._postprocess_images(mosaic_sub_images)
        mosaic_boxes = self._postprocess_boxes(mosaic_sub_boxes,
                                               tf.shape(boxes)[1])
        return mosaic_image, mosaic_boxes

    def _preprocess_boxes(self, boxes, max_obj_num):
        boxes_list = []
        obj_count = 0
        for box in boxes:
            get_idx = tf.where(
                tf.math.reduce_all(tf.math.is_finite(box), axis=(1, 2)))
            if get_idx.shape[0] == 0:
                warn_mess = f"No obj"
                logging.warning(warn_mess)
                boxes_list.append(None)
                continue
            bool_idx = tf.squeeze(get_idx, axis=1)
            obj_count += bool_idx.shape[0]
            if obj_count > max_obj_num:
                warn_mess = f"Object number exceed the upper limit: {max_obj_num}, return normal image"
                logging.warning(warn_mess)
                return None
            box = tf.cast(tf.gather(box, bool_idx), tf.int32)
            boxes_list.append(box)
        return boxes_list

    def _mosaic_divide_points(self) -> Tuple[int, int]:
        """Returns a  tuple of y and x which corresponds to mosaic divide points."""
        y_point = tf.random.uniform(
            shape=[1],
            minval=tf.cast(
                self._out_size[0] * (self._minimum_mosaic_image_dim / 100),
                tf.int32),
            maxval=tf.cast(
                self._out_size[0] *
                ((100 - self._minimum_mosaic_image_dim) / 100),
                tf.int32,
            ),
            dtype=tf.int32,
        )
        x_point = tf.random.uniform(
            shape=[1],
            minval=tf.cast(
                self._out_size[1] * (self._minimum_mosaic_image_dim / 100),
                tf.int32),
            maxval=tf.cast(
                self._out_size[1] *
                ((100 - self._minimum_mosaic_image_dim) / 100),
                tf.int32,
            ),
            dtype=tf.int32,
        )
        # print('y, x', y_point, x_point)
        return y_point, x_point

    def _mosaic(self, images, boxes, mosaic_divide_points):
        """Builds mosaic of provided images.
        Args:
        images: original single images to make mosaic.
        boxes: corresponding bounding boxes to images. Nx2x3
        mosaic_divide_points: Points to build mosaic around on given output size.
        Returns:
        A tuple of mosaic Image, Mosaic Boxes merged.
        """
        (
            mosaic_image_topleft,
            mosaic_image_topright,
            mosaic_image_bottomleft,
            mosaic_image_bottomright,
        ) = self._process_images(images, mosaic_divide_points)

        # Scale Boxes for TOP LEFT image.
        # Map_fn is replace with vectorized_map below for optimization purpose.
        if boxes[0] is None:
            mosaic_box_topleft = None
        else:
            mosaic_box_topleft = tf.vectorized_map(
                functools.partial(self._process_box,
                                  image=images[0],
                                  mosaic_image=mosaic_image_topleft),
                boxes[0],
            )

        # Scale and Pad Boxes for TOP RIGHT image.
        if boxes[1] is None:
            mosaic_box_topright = None
        else:
            mosaic_box_topright = tf.vectorized_map(
                functools.partial(self._process_box,
                                  image=images[1],
                                  mosaic_image=mosaic_image_topright),
                boxes[1],
            )
            transpose_topright = tf.transpose(mosaic_box_topright, [2, 0, 1])
            idx_tr = tf.constant([[1]])
            update_tr = tf.constant(
                tf.shape(mosaic_image_topleft)[1], tf.int32,
                [1] + list(tf.shape(transpose_topright[0])))
            transpose_topright = tf.tensor_scatter_nd_add(
                transpose_topright, idx_tr, update_tr)
            mosaic_box_topright = tf.transpose(transpose_topright, [1, 2, 0])

        # Scale and Pad Boxes for BOTTOM LEFT image.
        if boxes[2] is None:
            mosaic_box_bottomleft = None
        else:
            mosaic_box_bottomleft = tf.vectorized_map(
                functools.partial(self._process_box,
                                  image=images[2],
                                  mosaic_image=mosaic_image_bottomleft),
                boxes[2],
            )
            transpose_bottomleft = tf.transpose(mosaic_box_bottomleft,
                                                [2, 0, 1])
            idx_bl = tf.constant([[0]])
            update_bl = tf.constant(
                tf.shape(mosaic_image_topleft)[0], tf.int32,
                [1] + list(tf.shape(transpose_bottomleft[0])))
            transpose_bottomleft = tf.tensor_scatter_nd_add(
                transpose_bottomleft, idx_bl, update_bl)
            mosaic_box_bottomleft = tf.transpose(transpose_bottomleft,
                                                 [1, 2, 0])

        # Scale and Pad Boxes for BOTTOM RIGHT image.
        if boxes[3] is None:
            mosaic_box_bottomright = None
        else:
            mosaic_box_bottomright = tf.vectorized_map(
                functools.partial(self._process_box,
                                  image=images[3],
                                  mosaic_image=mosaic_image_bottomright),
                boxes[3],
            )
            transpose_bottomright = tf.transpose(mosaic_box_bottomright,
                                                 [2, 0, 1])
            idx_br = tf.constant([[0], [1]])
            update_br = tf.concat([
                tf.constant(
                    tf.shape(mosaic_image_topleft)[0], tf.int32,
                    [1] + list(tf.shape(transpose_bottomright[0]))),
                tf.constant(
                    tf.shape(mosaic_image_topleft)[1], tf.int32,
                    [1] + list(tf.shape(transpose_bottomright[0])))
            ],
                                  axis=0)
            transpose_bottomright = tf.tensor_scatter_nd_add(
                transpose_bottomright, idx_br, update_br)
            mosaic_box_bottomright = tf.transpose(transpose_bottomright,
                                                  [1, 2, 0])

        # Gather mosaic_sub_images and boxes.
        mosaic_images = [
            mosaic_image_topleft,
            mosaic_image_topright,
            mosaic_image_bottomleft,
            mosaic_image_bottomright,
        ]
        mosaic_boxes = [
            mosaic_box_topleft,
            mosaic_box_topright,
            mosaic_box_bottomleft,
            mosaic_box_bottomright,
        ]
        return mosaic_images, mosaic_boxes

    def _process_box(self, box, image, mosaic_image):
        """scale boxes with mosaic sub image.
        Args:
        box: mosaic image box.
        image: original image.
        mosaic_image: mosaic sub image.
        Returns:
        Scaled bounding boxes. 2x3
        """
        if self._process_type == "mosaic_scale":
            box_y = tf.cast(
                box[:, 0] * tf.shape(mosaic_image)[0] / tf.shape(image)[0],
                tf.int32)
            box_x = tf.cast(
                box[:, 1] * tf.shape(mosaic_image)[1] / tf.shape(image)[1],
                tf.int32)
            box = tf.concat([
                tf.expand_dims(box_y, axis=-1),
                tf.expand_dims(box_x, axis=-1),
                tf.expand_dims(box[:, 2], axis=-1)
            ],
                            axis=-1)
        return box

    def _process_images(self, images, mosaic_divide_points) -> Tuple:
        """Scale Sub Images.
        Args:
        images: original single images to make mosaic.
        mosaic_divide_points: Points to build mosaic around on given output.
        Returns:
        A tuple of scaled Mosaic sub images.
        """
        y, x = mosaic_divide_points[0][0], mosaic_divide_points[1][0]
        # print("-"*20, "_process_images:", mosaic_divide_points, y, x, tf.shape(images))
        if self._process_type == "mosaic_scale":
            mosaic_image_topleft = tf.image.resize(images[0], (y, x))
            mosaic_image_topright = tf.image.resize(images[1],
                                                    (y, self._out_size[1] - x))
            mosaic_image_bottomleft = tf.image.resize(
                images[2], (self._out_size[0] - y, x))
            mosaic_image_bottomright = tf.image.resize(
                images[3], (self._out_size[0] - y, self._out_size[1] - x))
            # print("-"*20, "_process_images", tf.shape(mosaic_image_topleft), tf.shape(mosaic_image_topright), tf.shape(mosaic_image_bottomleft), tf.shape(mosaic_image_bottomright))
        return (tf.cast(mosaic_image_topleft,
                        tf.uint8), tf.cast(mosaic_image_topright, tf.uint8),
                tf.cast(mosaic_image_bottomleft,
                        tf.uint8), tf.cast(mosaic_image_bottomright, tf.uint8))

    def _postprocess_boxes(self, mosaic_sub_boxes, max_obj_num):
        collect_boxes = []
        for mosaic_sub in mosaic_sub_boxes:
            if mosaic_sub is not None:
                collect_boxes.append(mosaic_sub)
        mosaic_boxes = tf.concat(collect_boxes, axis=0)
        mosaic_boxes = tf.cast(mosaic_boxes, dtype=tf.float32)
        if tf.shape(mosaic_boxes)[0] < max_obj_num:
            inf_boxes = tf.constant(
                [np.inf],
                dtype=tf.float32,
                shape=[max_obj_num - tf.shape(mosaic_boxes)[0]] +
                list(tf.shape(mosaic_boxes)[1:]))
            mosaic_boxes = tf.concat([mosaic_boxes, inf_boxes], axis=0)
        return mosaic_boxes

    def _postprocess_images(self, mosaic_sub_images):
        upper_stack = tf.concat([mosaic_sub_images[0], mosaic_sub_images[1]],
                                axis=1)
        lower_stack = tf.concat([mosaic_sub_images[2], mosaic_sub_images[3]],
                                axis=1)
        mosaic_image = tf.concat([upper_stack, lower_stack], axis=0)
        return mosaic_image