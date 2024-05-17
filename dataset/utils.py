import tensorflow as tf
import numpy as np


def gaussian_radius(det_size, min_overlap=0.6):
    """
        The function of gaussian_radius is determined the gaussian radius. They set the target min_overlap = 0.7 and derive
        from the corner tl and br point. The radius r1, r2, and r3 describe how far the corners can be offset to still fulfill min_overlap.
        There are three cases to generate radius.
        We assumed that the presumed box and g.t. box is overlapping in the initial state.
        1.  The presumed box shift to the upper left or the buttom right of g.t. box.
            Condition: The g.t box tl as center point draw a circle which externally cut on outer presumed box lines.
            and vice versa for br points. The equation looks like min_overlap = (h-r)*(w-r) / ((h-r)*(w-r)+2h*w) solving the quadratic function of radius.
        2.  The presumed box scale down and g.t. enclose presumed box.
            Condition: The presumed box tl as center point draw a circle which externally cut on outer g.t. box lines.
            and vice versa for br points. The equation looks like min_overlap = (h-2r)*(w-2r)/h*w solving the quadratic function of radius.
        3.  The presumed box scale up and presumed enclose g.t. box.
            Condition: The g.t. tl as center point draw a circle which externally cut on outer presumed box lines
            and vice versa for br points. The equation looks like min_overlap = h*w/(h+2r)(w+2r) solving the quadratic function of radius.
        Arguments:
            det_size(input type=tuple) - - in g.t. object's height and width
        Returns:
            r(Output type=float64)-- the minimum radius value for gaussian
        """
    height, width = det_size[..., 0], det_size[..., 1]
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = tf.math.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = tf.math.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = tf.math.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    R = tf.concat(
        [r1[..., tf.newaxis], r2[..., tf.newaxis], r3[..., tf.newaxis]],
        axis=-1)
    R = tf.where(tf.math.is_nan(R), np.inf, R)
    min_radius = tf.math.reduce_min(R, axis=-1)
    return tf.math.maximum(1., min_radius)


def draw_msra_gaussian(heatmap, center, sigma):
    """
        The function of draw_msra_gaussian use unnormalized 2D gaussian method to draw the heat map.
        It assigned the values around the g.t. key points for gaussian distribution.
        Arguments:
            heatmap  -- the heat map's height and width are down size four times from the image's shape
            center   -- the values of x and y
            sigma    -- gaussian's sigma
        Returns:
            heatmap -- the gaussian distribution values
        """

    tmp_size = sigma * 3
    mu_y = center[0]
    mu_x = center[1]
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    # diameter
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]

    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)

    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]], g[g_y[0]:g_y[1],
                                                         g_x[0]:g_x[1]])
    return heatmap


def gaussian_idxs(batch_size, max_obj_num, b_sigmas, b_coors):
    b_sub_idxs = b_coors[:, :, 0, :]
    b_obj_idxs = b_coors[:, :, 1, :]

    b_sub_idxs = tf.where(b_sub_idxs == np.inf, 0., b_sub_idxs)
    b_mask = tf.where(b_coors[:, :, 1, :] == np.inf, 0., 1.)

    b_sub_idxs = b_sub_idxs * b_mask
    b_sub_idxs = tf.where(b_sub_idxs == 0., np.inf, b_sub_idxs)

    b_sub_idxs = tf.expand_dims(b_sub_idxs, axis=-2)
    b_obj_idxs = tf.expand_dims(b_obj_idxs, axis=-2)
    rel_embed_idxs = tf.concat([b_sub_idxs, b_obj_idxs], axis=-2)

    b_sigmas = tf.tile(b_sigmas[..., None, None], [1, 1, 2, 2])
    random_idxs = tf.random.normal(shape=(batch_size, max_obj_num, 2, 2),
                                   mean=0.0,
                                   stddev=b_sigmas / 3,
                                   dtype=tf.dtypes.float32)
    rel_embed_idxs = random_idxs + rel_embed_idxs
    rel_embed_idxs = tf.where(tf.math.is_nan(rel_embed_idxs), np.inf,
                              rel_embed_idxs)
    rel_embed_idxs = tf.cast(rel_embed_idxs, tf.int32)
    rel_embed_idxs = tf.cast(rel_embed_idxs, tf.float32)
    rel_embed_idxs = tf.where(rel_embed_idxs < -1e8, np.inf, rel_embed_idxs)
    rel_embed_idxs = tf.where(rel_embed_idxs > 1e8, np.inf, rel_embed_idxs)
    return rel_embed_idxs


def gen_bboxes(batch_size, b_coors, b_cates, max_obj_num, image_input_sizes):
    b_center_kps = b_coors[:, :, 0, :]
    finite_mask = tf.math.is_finite(b_center_kps)[:, :, 0]
    finite_mask = tf.where(finite_mask == True, 1., 0.)
    num_bbox = tf.math.reduce_sum(finite_mask, axis=-1, keepdims=True)
    b_bboxes = b_coors[:, :, 1:3, :]
    # b_bboxes = tf.einsum('b n c d, b d -> b n c d', b_bboxes,
    #                      1 / image_input_sizes)
    b_bboxes = tf.reshape(b_bboxes, [batch_size, -1, 4])
    b_bboxes = tf.where(tf.math.is_nan(b_bboxes), -1., b_bboxes)
    b_bboxes = tf.where(tf.math.is_inf(b_bboxes), -1., b_bboxes)
    b_bboxes = tf.where(b_bboxes == -1., np.inf, b_bboxes)
    return b_bboxes, b_cates, num_bbox


def gen_landmarks(batch_size, max_obj_num, b_coords, schema_type):
    """
        landmarks link[https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup-768x619.jpg]
        profile face: 0 ~ 16
        eye_brows: 17 ~ 26
        eyes: 27 ~ 38
        nose: 39 ~ 47
        outer mouth: 48 ~ 59
        inner mouth: 60 ~ 68 
    """

    if schema_type == 25:
        lnmk_25_schema = tf.constant([
            0, 8, 16, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 42,
            48, 50, 51, 52, 54, 56, 57, 58
        ],
                                     shape=(25, 1),
                                     dtype=tf.int32)
        # (96, 15, 25, 2)
        b_coords = tf.transpose(b_coords, [2, 0, 1, 3])
        b_coords = tf.gather_nd(b_coords, lnmk_25_schema)
        b_coords = tf.transpose(b_coords, [1, 2, 0, 3])
    return b_coords


def xyxy2xywh(b_center, b_obj_sizes):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    xywh = tf.concat([b_center[:, :, None, :], b_obj_sizes[:, :, None, :]],
                     axis=-2)
    return xywh