import tensorflow as tf
import numpy as np
import cv2
import random
import copy
from functools import partial
from tensorpack.dataflow import *
from albumentations import Compose, CoarseDropout, GridDropout
from pprint import pprint


class Base:

    def __init__(self):
        self.clc_aug_funcs = {
            "bright": partial(tf.image.random_brightness, max_delta=0.1),
            "saturat": partial(tf.image.random_saturation,
                               lower=0.4,
                               upper=1.8),
            "hue": partial(tf.image.random_hue, max_delta=0.1),
            "contrast": partial(tf.image.random_contrast, lower=0.6, upper=1.4)
        }

    def color_aug(self, b_imgs, aug_chains):
        if not aug_chains or len(aug_chains) == 0:
            return b_imgs
        for aug_name in self.config.augments.color_chains:
            b_imgs = self.clc_aug_funcs[aug_name](b_imgs)
        return b_imgs

    def album_augs(self, album_config, b_imgs):

        def album_aug(images):
            aug_imgs = np.asarray(
                list(map(lambda x: transforms(image=x)["image"],
                         images))).astype(np.uint8)
            return aug_imgs

        transform_list = []
        for aug_name in album_config:
            if aug_name == "gridmask":
                transform_list.append(
                    GridDropout(
                        ratio=album_config.gridmask.ratio,
                        unit_size_min=album_config.gridmask.unit_size_min,
                        unit_size_max=album_config.gridmask.unit_size_max,
                        random_offset=album_config.gridmask.random_offset))
            if aug_name == "cutout":
                transform_list.append(
                    CoarseDropout(max_holes=album_config.cutout.max_holes,
                                  max_height=album_config.cutout.max_height,
                                  max_width=album_config.cutout.max_width,
                                  min_holes=album_config.cutout.min_holes,
                                  min_height=album_config.cutout.min_height,
                                  min_width=album_config.cutout.min_width,
                                  fill_value=album_config.cutout.fill_value))
            transforms = Compose(transform_list)
        b_imgs = tf.numpy_function(func=album_aug, inp=[b_imgs], Tout=tf.uint8)
        return b_imgs

    def tensorpack_augs(self, b_labels, b_imgs, task, flip_probs, max_obj_num,
                        tensorpack_chains):
        f'''
            Do random ratation, crop and resize by using "tensorpack" augmentation class.
                https://github.com/tensorpack/tensorpack
            1. resize the annotation to be the same as original image size.
            2. rotating at the image center by small degree.
            3. correct the minus points due to rotation by function correct_out_point.
            4. cropping ratio is 0.8.
            5. resize back.
            return :  B, N, [tl, br], [y, x, c] [B, N, C, D]
        '''
        # preprocess for different task annos
        self.task = str(task.numpy())
        b_labels = b_labels.numpy().astype(np.float32)
        b_imgs = b_imgs.numpy().astype(np.uint8)
        flip_probs = flip_probs.numpy()
        _, h, w, c = b_imgs.shape
        aug_prob = 0.4
        max_obj_num = int(max_obj_num.numpy())
        tmp_imgs, tmp_labels = [], []
        for img, labels, flip_prob in zip(b_imgs, b_labels, flip_probs):
            valid_mask = np.all(np.isfinite(labels), axis=-1)
            labels = labels[valid_mask]
            annos = labels[..., 2:]
            annos = annos.reshape([-1, 2, 2])
            cates = labels[..., :2]
            if flip_prob:
                annos = self._flip(img, annos, w)
            annos = annos.reshape([-1, 4])
            labels = np.concatenate([cates, annos], axis=-1)
            for tensorpack_aug in tensorpack_chains:
                if tensorpack_aug == "CropTransform":
                    # do crop transform
                    crop_param = random.random()
                    if crop_param < aug_prob:
                        img, coors, cates = self.crop_transform(
                            img, coors, h, w)
                        coors = np.concatenate([coors, cates], axis=-1)
                elif tensorpack_aug == "RandomPaste":
                    # do random paste
                    if random.random() < aug_prob:
                        img, coors, cates = self.random_paste(img, coors, h, w)
                        coors = np.concatenate([coors, cates], axis=-1)

                elif tensorpack_aug == "WarpAffineTransform":
                    if random.random() < aug_prob:

                        img, coors, cates = self.warp_affine_transform(
                            img, coors, h, w)
                        coors = np.concatenate([coors, cates], axis=-1)
                elif tensorpack_aug == "RandomSquareCrop":
                    if random.random() < aug_prob:
                        img, coors, cates = self.random_square_crop(
                            img, coors, h, w)
                        coors = np.concatenate([coors, cates], axis=-1)
            tmp_imgs.append(img)
            # coors = np.concatenate([annos, coors[..., -1:]], axis=-1)
            n, c = labels.shape
            if max_obj_num - n > 0:
                complement = np.empty([max_obj_num - n, c])
                complement.fill(np.inf)
                complement = complement.astype(np.float32)
                labels = np.concatenate([labels, complement], axis=0)
            tmp_labels.append(labels)

        tmp_labels = np.asarray(tmp_labels)
        b_labels = np.stack(tmp_labels).astype(np.float32)
        b_imgs = np.stack(tmp_imgs)
        return b_imgs, b_labels

    def _flip(self, img, objs_kps, w):

        n, c, d = objs_kps.shape
        objs_wilds = objs_kps[:, 1, 0] - objs_kps[:, 0, 0]
        objs_wilds = np.tile(objs_wilds[:, np.newaxis, np.newaxis], [1, c, 1])
        objs_kps_x = objs_kps[..., :1]
        objs_kps_x = -objs_kps_x + w - 1
        objs_kps_y = objs_kps[..., -1:]
        objs_kps_x[:, 0, :] -= objs_wilds[:, 0, :]
        objs_kps_x[:, 1, :] += objs_wilds[:, 1, :]
        objs_kps = np.concatenate([objs_kps_x, objs_kps_y], axis=-1)
        if 'obj_det' in self.task:
            return objs_kps

        objs_boxes = objs_kps[:, :2]
        jawline = objs_kps[:, 2:19][:, ::-1]
        eyebrows = objs_kps[:, 19:29][:, ::-1]
        L_eyes = objs_kps[:, 29:35][:, [3, 2, 1, 0, 5, 4]]

        R_eyes = objs_kps[:, 35:41][:, [3, 2, 1, 0, 5, 4]]
        nose = objs_kps[:, 41:50][:, [0, 1, 2, 3, 8, 7, 6, 5, 4]]
        out_lips = objs_kps[:, 50:62][:,
                                      [6, 5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7]]

        in_lips = objs_kps[:, 62:70][:, [4, 3, 2, 1, 0, 7, 6, 5]]

        objs_kps = np.concatenate([
            objs_boxes, jawline, eyebrows, R_eyes, L_eyes, nose, out_lips,
            in_lips
        ],
                                  axis=-2)
        return objs_kps

    def draw_mask(self, img, coors):
        iris_mask = np.zeros(shape=img.shape, dtype=np.uint8)
        eyelid_mask = np.zeros(shape=img.shape, dtype=np.uint8)

        valid_mask = np.isfinite(coors)[:, 0, 0]
        coors = (coors[valid_mask][..., :2]).astype(np.int32)
        eyelid_coors = (coors[:, :-8, ::-1]).astype(np.int32)

        eyelid_coors = np.concatenate(
            [eyelid_coors[:, :17], eyelid_coors[:, 17:][:, ::-1, :]], axis=1)

        iris_coors = coors[:, -8:, ::-1]
        if np.all(iris_coors == 0.):
            mask = np.asarray(cv2.fillPoly(eyelid_mask, eyelid_coors,
                                           255)).astype(np.int32)
        else:
            mask = np.asarray(cv2.fillPoly(
                eyelid_mask, eyelid_coors, 255)).astype(np.int32) + np.asarray(
                    cv2.fillPoly(iris_mask, iris_coors, 255)).astype(np.int32)

        mask[mask <= 255] = 0
        mask = mask.astype(np.uint8)

        if random.random() < 0.5:
            rgb_gray_iris = np.random.randint(low=1,
                                              high=70,
                                              size=mask[mask > 128].shape)
            img[mask > 128] = rgb_gray_iris

        bg_mask = 255 * np.ones_like(mask) - mask
        mask = np.concatenate([mask, bg_mask], axis=-1)
        return img, mask

    def crop_transform(self, img, coors, h, w):
        annos, cates = coors[..., :-1], coors[..., -1:]
        if random.random() < 0.8:
            cropped_tl, cropped_br = annos[0, 0, :].astype(
                np.int32), annos[0, 1, :].astype(np.int32)
            ori_shape = np.asarray(img.shape[:2])
            img = img[cropped_tl[1]:cropped_br[1],
                      cropped_tl[0]:cropped_br[0], :]
            annos = annos - cropped_tl
            h, w, _ = img.shape
            annos = np.einsum('n c d, d -> n c d', annos,
                              ori_shape / np.array([w, h]))
            img = cv2.resize(img, tuple(ori_shape[::-1]))
            return img, annos, cates
        base_ratio = 0.8
        # crop_ratio = np.random.randint(low=5, high=20) / 100.0
        x_crop_ratio = base_ratio + np.random.randint(low=5, high=20) / 100.0
        y_crop_ratio = base_ratio + np.random.randint(low=5, high=20) / 100.0
        h_crop, w_crop = int(round(h * y_crop_ratio)), int(
            round(w * x_crop_ratio))
        h1, w1 = np.random.randint(low=0, high=h - h_crop +
                                   1), np.random.randint(low=0,
                                                         high=w - w_crop + 1)
        crop_transform = CropTransform(h1, w1, h_crop, w_crop)
        img_out = crop_transform.apply_image(img)
        # N, C, D
        origin_annos = copy.deepcopy(annos)
        origin_cates = copy.deepcopy(cates)
        annos_out = crop_transform.apply_coords(annos)
        resize_transform = ResizeTransform(h_crop, w_crop, h, w,
                                           cv2.INTER_CUBIC)
        img_out = resize_transform.apply_image(img_out)
        annos_out = resize_transform.apply_coords(annos_out)
        annos_out = self.correct_out_point(annos_out, cates, 0, 0, h, w)
        if annos_out.any():
            return img_out, annos_out[..., :-1], annos_out[..., -1:]
        else:
            return img, origin_annos, origin_cates

    def random_paste(self, img, coors, h, w):
        annos, cates = coors[..., :-1], coors[..., -1:]
        base_ratio = 1.0
        bg_ratio = base_ratio + np.random.random_sample() * 0.4
        bg_h, bg_w = round(h * bg_ratio), round(w * bg_ratio)
        obj = RandomPasetWithMeanBackground((bg_h, bg_w))
        l = obj.get_transform(img)
        if l == False:
            return img, annos, cates
        img_out = obj._impl(img, l)
        annos_out = annos + l

        if annos_out.any():
            resize_transform = ResizeTransform(bg_h, bg_w, h, w,
                                               cv2.INTER_CUBIC)
            img_out = resize_transform.apply_image(img_out)
            annos_out = resize_transform.apply_coords(annos_out)
            annos_out = self.correct_out_point(annos_out, cates, 0, 0, h, w)
            if annos_out.any():
                return img_out, annos_out[..., :2], annos_out[..., -1:]
            else:
                return img, annos, cates
        else:
            return img, annos, cates

    def warp_affine_transform(self, img, coors, h, w):
        annos, cates = coors[..., :-1], coors[..., -1:]
        img_center = (w / 2, h / 2)
        rotation_angle = np.random.randint(low=-15, high=15)
        mat = cv2.getRotationMatrix2D(img_center, rotation_angle, 1)
        affine = WarpAffineTransform(mat, (w, h))
        img_out = affine.apply_image(img)

        annos_out = affine.apply_coords(annos)

        annos_out = self.correct_out_point(annos_out, cates, 0, 0, h, w)

        if annos_out.any():
            return img_out, annos_out[..., :2], annos_out[..., -1:]
        else:
            return img, annos, cates

    def correct_out_point(self, annos, cates, h1, w1, h2, w2):

        def gen_boolean_mask(check):
            check = np.all(check, axis=-1)
            return check

        if np.all(annos > 0.) or 'tdmm' in self.task:
            annos = np.concatenate([annos, cates], axis=-1)
            return annos
        else:
            return np.array([])

        valid_indice_x = np.where(annos[..., 0] < w1)
        valid_indice_y = np.where(annos[..., 1] < h1)
        annos[:, :, 0][valid_indice_x] = w1
        annos[:, :, 1][valid_indice_y] = h1
        valid_indice_x = np.where(annos[..., 0] > w2)
        valid_indice_y = np.where(annos[..., 1] > h2)
        annos[:, :, 0][valid_indice_x] = w2 - 1
        annos[:, :, 1][valid_indice_y] = h2 - 1
        annos = np.concatenate([annos, cates], axis=-1)

        _, _, c = annos.shape
        axis_check = annos[:, 0, :2] != annos[:, 1, :2]
        if np.any(axis_check == False):
            axis_check = gen_boolean_mask(axis_check)
        annos = annos[axis_check]
        annos = annos.reshape((-1, 2, c))
        if len(annos) > 0:
            height_width = annos[:, 1, :2] - annos[:, 0, :2]
            area_mask = height_width[:, 0] * height_width[:, 1] > 10
            annos = annos[area_mask]
            return annos
        else:
            return annos

    def random_square_crop(self, img, coors, h, w):
        crop_choice = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        max_scale = np.amax(crop_choice)
        scale_retry = 0
        boxes = coors[:, :2, :2].reshape([-1, 4])
        lnmks = coors[:, 2:, :2]
        cates = coors[..., -1:]

        bbox_clip_border = True
        while True:
            scale_retry += 1
            if scale_retry == 1 or max_scale > 1.0:
                scale = np.random.choice(crop_choice)
            else:
                #scale = min(scale*1.2, max_scale)
                scale = scale * 1.2
            for i in range(250):
                short_side = min(w, h)
                cw = int(scale * short_side)
                ch = cw
                # TODO +1
                if w == cw:
                    left = 0
                elif w > cw:
                    #left = random.uniform(w - cw)
                    left = random.randint(0, w - cw)
                else:
                    left = random.randint(w - cw, 0)
                if h == ch:
                    top = 0
                elif h > ch:
                    #top = random.uniform(h - ch)
                    top = random.randint(0, h - ch)
                else:
                    top = random.randint(h - ch, 0)

                patch = np.array(
                    (int(left), int(top), int(left + cw), int(top + ch)),
                    dtype=np.int32)

                def is_center_of_bboxes_in_patch(boxes, patch):
                    # TODO >=
                    center = (boxes[:, :2] + boxes[:, 2:]) / 2
                    mask = ((center[:, 0] > patch[0]) *
                            (center[:, 1] > patch[1]) *
                            (center[:, 0] < patch[2]) *
                            (center[:, 1] < patch[3]))
                    return mask

                mask = is_center_of_bboxes_in_patch(boxes, patch)
                if not mask.any():
                    continue
                #print('BBB', key, boxes.shape)
                # mask = is_center_of_bboxes_in_patch(boxes, patch)
                boxes = boxes[mask]
                if bbox_clip_border:
                    boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                    boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                #print('AAAA', kps_key, keypointss.shape, mask.shape)
                lnmks = lnmks[mask, :, :]
                if bbox_clip_border:
                    lnmks[:, :, :2] = lnmks[:, :, :2].clip(max=patch[2:])
                    lnmks[:, :, :2] = lnmks[:, :, :2].clip(min=patch[:2])
                lnmks[:, :, 0] -= patch[0]
                lnmks[:, :, 1] -= patch[1]
                rimg = np.ones((ch, cw, 3), dtype=img.dtype) * 128
                patch_from = patch.copy()
                patch_from[0] = max(0, patch_from[0])
                patch_from[1] = max(0, patch_from[1])
                patch_from[2] = min(img.shape[1], patch_from[2])
                patch_from[3] = min(img.shape[0], patch_from[3])
                patch_to = patch.copy()
                patch_to[0] = max(0, patch_to[0] * -1)
                patch_to[1] = max(0, patch_to[1] * -1)
                patch_to[2] = patch_to[0] + (patch_from[2] - patch_from[0])
                patch_to[3] = patch_to[1] + (patch_from[3] - patch_from[1])
                rimg[patch_to[1]:patch_to[3],
                     patch_to[0]:patch_to[2], :] = img[
                         patch_from[1]:patch_from[3],
                         patch_from[0]:patch_from[2], :]
                #print(img.shape, scale, patch, patch_from, patch_to, rimg.shape)
                img = rimg
                aug_h, aug_w = img.shape[:2]
                img = cv2.resize(img, (h, w))
                resized_ratio = np.array([w, h]) / np.array([aug_w, aug_h])
                boxes = boxes.reshape([-1, 2, 2])
                annos = np.concatenate([boxes, lnmks], axis=-2)
                annos = np.einsum('n c d, d -> n c d', annos, resized_ratio)
                return img, annos, cates


class RandomPasetWithMeanBackground(imgaug.RandomPaste):

    def get_transform(self, img):
        img_shape = img.shape[:2]
        if self.background_shape[0] > img_shape[0] and self.background_shape[
                1] > img_shape[1]:
            y0 = np.random.randint(self.background_shape[0] - img_shape[0])
            x0 = np.random.randint(self.background_shape[1] - img_shape[1])
            l = int(x0), int(y0)
            return l
        else:
            return False

    def _impl(self, img, loc):
        x0, y0 = loc
        img_shape = img.shape[:2]
        self.background_shape = np.asarray(self.background_shape).astype(
            np.int32)
        self.background_shape = tuple(self.background_shape)
        background = self.background_filler.fill(self.background_shape, img)
        image_mean = img.mean(axis=(0, 1))
        background[:, :] = image_mean
        background[y0:y0 + img_shape[0], x0:x0 + img_shape[1]] = img
        return background


class WarpAffineTransform:

    def __init__(self,
                 mat,
                 dsize,
                 interp=cv2.INTER_LINEAR,
                 borderMode=cv2.BORDER_CONSTANT,
                 borderValue=0):
        self.mat = mat
        self.dsize = dsize
        self.interp = interp
        self.borderMode = borderMode
        self.borderValue = borderValue

    def apply_image(self, img):
        ret = cv2.warpAffine(img,
                             self.mat,
                             self.dsize,
                             flags=self.interp,
                             borderMode=self.borderMode,
                             borderValue=self.borderValue)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords):

        n, c, d = coords.shape
        expand_ones = np.ones((n, c, 1), dtype='f4')
        coords = np.concatenate((coords, expand_ones), axis=-1)
        rotate_matrices = self.mat.T
        # n, d = coords.shape
        # expand_ones = np.ones((n, 1), dtype='f4')
        # coords = np.concatenate((coords, expand_ones), axis=-1)
        # rotate_matrices = self.mat.T
        coords = np.dot(coords, rotate_matrices)
        return coords


class CropTransform(imgaug.Transform):
    """
    Crop a subimage from an image.
    """

    def __init__(self, y0, x0, h, w):
        super(CropTransform, self).__init__()
        self._init(locals())

    def apply_image(self, img):
        return img[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w]

    def apply_coords(self, coords):
        coords[:, :, 0] -= self.x0
        coords[:, :, 1] -= self.y0
        return coords


class ResizeTransform(imgaug.Transform):
    """
    Resize the image.
    """

    def __init__(self, h, w, new_h, new_w, interp):
        """
        Args:
            h, w (int):
            new_h, new_w (int):
            interp (int): cv2 interpolation method
        """
        super(ResizeTransform, self).__init__()
        self._init(locals())

    def apply_image(self, img):
        assert img.shape[:2] == (self.h, self.w)
        ret = cv2.resize(img, (self.new_w, self.new_h),
                         interpolation=self.interp)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords):
        coords[..., 0] = coords[..., 0] * (self.new_w * 1.0 / self.w)
        coords[..., 1] = coords[..., 1] * (self.new_h * 1.0 / self.h)
        return coords
