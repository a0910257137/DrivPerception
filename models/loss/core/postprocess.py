import numpy as np
import tensorflow as tf


def build_targets(anchors, feat_sizes, head_model, predictions, targets):
    '''
    predictions
    [16, 3, 32, 32, 85]
    [16, 3, 16, 16, 85]
    [16, 3, 8, 8, 85]
    torch.tensor(predictions[i].shape)[[3, 2, 3, 2]]
    [32,32,32,32]
    [16,16,16,16]
    [8,8,8,8]
    targets[3,x,7]
    t [index, class, x, y, w, h, head_index]
    '''

    def true_nt(t, anchor, feat_size):
        # Matches
        r = t[:, :, 4:6] / anchor[:, None]  # wh ratio
        j = tf.math.reduce_max(tf.math.maximum(r, 1. / r),
                               axis=-1) < ANCHOR_THRESHOLD  # compare
        t = t[j]
        # Offsets
        gxy = t[:, 2:4]  # grid xy
        gxi = feat_size - gxy  # inverse
        jk = (gxy % 1. < g) & (gxy > 1.)
        lm = (gxi % 1. < g) & (gxi > 1.)
        j = tf.ones_like(jk[:, :-1], dtype=tf.bool)
        j = tf.transpose(tf.concat([j, jk, lm], axis=-1), perm=[1, 0])
        t = tf.tile(t[None, :, :], (5, 1, 1))[j]
        offsets = (tf.zeros_like(gxy)[None] + off[:, None])[j]
        return t, offsets

    def false_nt(t):
        t, offsets = targets[0], tf.cast(0., tf.float32)
        return t, offsets

    nl = head_model.nl
    na = head_model.na
    valid_mask = tf.reduce_all(tf.math.is_finite(targets), axis=-1)
    # valid_pos = tf.cast(valid_mask, tf.float32)
    targets = targets[valid_mask]
    nt = tf.shape(targets)[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = tf.ones(2, dtype=tf.float32)  # normalized to gridspace gain
    ai = tf.reshape(tf.range(na, dtype=tf.float32), [na, 1, 1])
    ai = tf.tile(ai, [1, nt, 1])  # same as .repeat_interleave(nt)

    targets = tf.concat((tf.tile(targets[None, :, :], [na, 1, 1]), ai),
                        axis=-1)  # append anchor indices

    g = 0.5  # bias
    ANCHOR_THRESHOLD = 4.0
    off = tf.constant(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],  # j,k,l,m
            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
        ],
        dtype=tf.float32) * g  # offsets
    feat_sizes = feat_sizes[:, ::-1]
    for i in range(nl):
        anchor = anchors[i]  #[3,2]
        feat_size = feat_sizes[i]
        gain = tf.concat([feat_size, feat_size], axis=0)
        # Match targets to anchors
        t = targets[..., 2:6] * gain
        t = tf.concat([targets[..., :2], t, targets[..., -1:]], axis=-1)
        t, offsets = tf.cond(nt > 0,
                             true_fn=lambda: true_nt(t, anchor, feat_size),
                             false_fn=lambda: false_nt(t))

        # if nt:
        #     # Matches
        #     r = t[:, :, 4:6] / anchor[:, None]  # wh ratio
        #     j = tf.math.reduce_max(tf.math.maximum(r, 1. / r),
        #                            axis=-1) < ANCHOR_THRESHOLD  # compare
        #     t = t[j]
        #     # Offsets
        #     gxy = t[:, 2:4]  # grid xy
        #     gxi = feat_size - gxy  # inverse
        #     jk = (gxy % 1. < g) & (gxy > 1.)
        #     lm = (gxi % 1. < g) & (gxi > 1.)
        #     j = tf.ones_like(jk[:, :-1], dtype=tf.bool)
        #     j = tf.transpose(tf.concat([j, jk, lm], axis=-1), perm=[1, 0])
        #     t = tf.tile(t[None, :, :], (5, 1, 1))[j]
        #     offsets = (tf.zeros_like(gxy)[None] + off[:, None])[j]
        # else:
        #     t = targets[0]
        #     offsets = 0
        # Define
        bc = tf.cast(t[:, :2], tf.int32)  # image, class
        b, c = bc[:, 0], bc[:, 1]  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = tf.cast((gxy - offsets), tf.int32)
        gi, gj = gij[:, 0], gij[:, 1]  # grid xy indices
        # Append
        a = tf.cast(t[:, 6], tf.int32)  # anchor indices
        indices.append(
            (b, a, tf.clip_by_value(gj, 0, tf.cast(gain[3] - 1, tf.int32)),
             tf.clip_by_value(gi, 0, tf.cast(
                 gain[2] - 1, tf.int32))))  # image, anchor, grid indices

        tbox.append(tf.concat((gxy - tf.cast(gij, tf.float32), gwh),
                              axis=1))  # box
        anchor = tf.tile(anchor[None, :, :], [tf.shape(a)[0], 1, 1])
        n_idx = tf.range(tf.shape(a)[0], dtype=tf.int32)
        a = tf.concat([n_idx[:, None], a[:, None]], axis=-1)
        anchor = tf.gather_nd(anchor, a)
        anch.append(anchor)  # anchor
        tcls.append(c)  # class
    return tcls, tbox, indices, anch
