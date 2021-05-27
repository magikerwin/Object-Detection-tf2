import tensorflow as tf
from algorithms.common.utils import _iou, prior_to_box

def _encode_bbox(matched, priors):
    """
    (x_min, y_min, x_max, y_min) --> encoded_xywh
    
    x_encoded = (x_min + x_max) / 2 - x_anchor
    y_encoded = (y_min + y_max) / 2 - y_anchor
    w_encoded = log((x_max - x_min) / w_anchor)
    h_encoded = log((y_max - y_min) / h_anchor)
    """
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = tf.math.log(g_wh)
    return tf.concat([g_cxcy, g_wh], 1) # [num_priors,4]

def _decode_bbox(pred, priors):
    """
    encoded_xywh --> (x_min, y_min, x_max, y_min)
    
    x = x_anchor + x_encoded
    y = y_anchor + y_encoded
    w = w_anchor * exp(w_encoded)
    h = h_anchor * exp(h_encoded)
    x_min = x - w / 2
    y_min = y - h / 2
    x_max = x + w / 2
    y_max = y + h / 2
    """
    cxcy = priors[:, :2] + pred[:, :2]
    wh = priors[:, 2:] * tf.math.exp(pred[:, 2:])
    return tf.concat([cxcy - wh / 2, cxcy + wh / 2], axis=1)

def get_encode_func(input_size, priors, match_thresh, ignore_thresh):
    assert ignore_thresh <= match_thresh
    
    def encode(bboxes):
    
        bboxes = tf.stack([bboxes[:, 0] / input_size[0], 
                           bboxes[:, 1] / input_size[1],
                           bboxes[:, 2] / input_size[0],
                           bboxes[:, 3] / input_size[1]], axis=-1)
        flattened_priors = tf.concat([tf.reshape(p, [-1, 4]) for p in priors], axis=0)
        overlaps = _iou(bboxes, prior_to_box(flattened_priors))

        # (Bipartite Matching)
        # [num_objects] best prior for each ground truth
        best_prior_overlap, best_prior_idx = tf.math.top_k(overlaps, k=1)
        best_prior_overlap = best_prior_overlap[:, 0]
        best_prior_idx = best_prior_idx[:, 0]

        # [num_priors] best ground truth for each prior
        overlaps_t = tf.transpose(overlaps)
        best_truth_overlap, best_truth_idx = tf.math.top_k(overlaps_t, k=1)
        best_truth_overlap = best_truth_overlap[:, 0]
        best_truth_idx = best_truth_idx[:, 0]

        # ensure best prior
        def _loop_body(i, bt_idx, bt_overlap):
            bp_mask = tf.one_hot(best_prior_idx[i], tf.shape(bt_idx)[0])
            bp_mask_int = tf.cast(bp_mask, tf.int32)
            new_bt_idx = bt_idx * (1 - bp_mask_int) + bp_mask_int * i
            bp_mask_float = tf.cast(bp_mask, tf.float32)
            new_bt_overlap = bt_overlap * (1 - bp_mask_float) + bp_mask_float * 2
            return tf.cond(best_prior_overlap[i] > match_thresh,
                           lambda: (i + 1, new_bt_idx, new_bt_overlap),
                           lambda: (i + 1, bt_idx, bt_overlap))
        _, best_truth_idx, best_truth_overlap = tf.while_loop(
            lambda i, bt_idx, bt_overlap: tf.less(i, tf.shape(best_prior_idx)[0]),
            _loop_body, [tf.constant(0), best_truth_idx, best_truth_overlap])

        matches_bboxes = tf.gather(bboxes, best_truth_idx)  # [num_priors, 4]

        # confidence (1: pos, 0: neg, -1: ignore)
        conf_t = tf.cast(best_truth_overlap > match_thresh, tf.float32)
        conf_t = tf.where(
            tf.logical_and(best_truth_overlap < match_thresh, best_truth_overlap > ignore_thresh),
            tf.ones_like(conf_t) * -1, conf_t)

        # localization
        loc_t = _encode_bbox(matches_bboxes, flattened_priors)

        return tf.concat([conf_t[..., tf.newaxis], loc_t], axis=-1)
    return encode

def decode(features, priors):
    flattened_priors = tf.concat([tf.reshape(p, [-1, 4]) for p in priors], axis=0)
    bboxes = _decode_bbox(features[:, 1:], flattened_priors)
    return tf.concat([features[:, 0:1], bboxes], axis=-1)