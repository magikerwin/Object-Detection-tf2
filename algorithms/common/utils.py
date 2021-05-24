import tensorflow as tf

def prior_box(input_size, steps, anchor_sizes):
    """prior box"""
    input_size = tf.convert_to_tensor(input_size, dtype=tf.float32)
    steps = tf.convert_to_tensor(steps, dtype=tf.float32)
    feature_maps = tf.math.ceil(input_size[tf.newaxis, ...] / steps[..., tf.newaxis])
    
    # sorting anchor sizes by area
    anchor_sizes = sorted(anchor_sizes, key=lambda x: x[0]*x[1])
    anchor_sizes = tf.reshape(tf.convert_to_tensor(anchor_sizes, dtype=tf.float32), [tf.shape(steps)[0], -1, 2])
    
    priors = []
    for idx, [h, w] in enumerate(feature_maps):
        num_anchors = len(anchor_sizes[idx])
        
        grid_x, grid_y = tf.meshgrid(tf.range(w), tf.range(h))
        cx = (grid_x + 0.5) * steps[idx] / input_size[1]
        cy = (grid_y + 0.5) * steps[idx] / input_size[0]
        cxcy = tf.stack([cx, cy], axis=-1)
        cxcy = tf.tile(tf.expand_dims(cxcy, 2), [1,1,num_anchors,1])
        
        sx = anchor_sizes[idx][:, 0] / input_size[1]
        sy = anchor_sizes[idx][:, 1] / input_size[0]
        sxsy = tf.stack([sx, sy], 1)
        sxsy = tf.tile(sxsy[tf.newaxis, tf.newaxis, ...], [h,w,1,1])
        
        priors.append(tf.concat([cxcy, sxsy], axis=-1))
        
    return priors

def prior_to_box(prior):
    """convert prior to (xmin, ymin xmax, ymax)"""    
    return tf.concat((prior[..., :2] - prior[..., 2:] / 2, 
                      prior[..., :2] + prior[..., 2:] / 2), axis=1)

def _intersect(bboxes_a, bboxes_b):
    """
    calculate intersection over union
    
    [num_bboxes_a, 2] --> [num_bboxes_a, 1, 2] --> [num_bboxes_a, num_bboxes_b, 2]
    [num_bboxes_b, 2] --> [num_bboxes_b, 2, 1] --> [num_bboxes_a, num_bboxes_b, 2]
    """
    
    num_bboxes_a = tf.shape(bboxes_a)[0]
    num_bboxes_b = tf.shape(bboxes_b)[0]
    right_bottom = tf.minimum(
        tf.broadcast_to(tf.expand_dims(bboxes_a[..., 2:], 1), [num_bboxes_a, num_bboxes_b, 2]),
        tf.broadcast_to(tf.expand_dims(bboxes_b[..., 2:], 0), [num_bboxes_a, num_bboxes_b, 2])
    )
    left_top = tf.maximum(
        tf.broadcast_to(tf.expand_dims(bboxes_a[..., :2], 1), [num_bboxes_a, num_bboxes_b, 2]),
        tf.broadcast_to(tf.expand_dims(bboxes_b[..., :2], 0), [num_bboxes_a, num_bboxes_b, 2])
    )    
    inter_wh = right_bottom - left_top
    inter_wh = tf.maximum(inter_wh, tf.zeros_like(inter_wh))
    return inter_wh[..., 0] * inter_wh[..., 1]

def _iou(bboxes_a, bboxes_b):
    """calculate intersection over union"""
    
    inter = _intersect(bboxes_a, bboxes_b)
    area_a = (bboxes_a[..., 2] - bboxes_a[..., 0]) * (bboxes_a[..., 3] - bboxes_a[..., 1])
    area_b = (bboxes_b[..., 2] - bboxes_b[..., 0]) * (bboxes_b[..., 3] - bboxes_b[..., 1])
    area_a = tf.broadcast_to(tf.expand_dims(area_a, 1), tf.shape(inter))
    area_b = tf.broadcast_to(tf.expand_dims(area_b, 0), tf.shape(inter))
    union = area_a + area_b - inter

    # avoid zero division
    union = tf.where(union != 0, union, 1)
    
    return inter / union