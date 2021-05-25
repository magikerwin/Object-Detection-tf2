import tensorflow as tf

def _smooth_l1_loss(y_true, y_pred):
    t = tf.abs(y_pred - y_true)
    return tf.where(t < 1, 0.5 * t ** 2, t - 0.5)

def _focal_loss(y_true, y_pred, alpha=1, gamma=2):
    return alpha * tf.pow(tf.abs(y_true - y_pred), gamma)

def get_losses(labels, preds, num_classes):
    """multi-box losses"""
    
    num_batchs = tf.shape(labels)[0]
    preds = [tf.reshape(pred, shape=[num_batchs, -1, (num_classes + 4)]) for pred in preds]
    preds = tf.concat(preds, axis=1)
    
    class_true = labels[..., :1]
    class_pred = preds[..., :1]
    loc_true = labels[..., 1:]
    loc_pred = preds[..., 1:]
    
    mask_pos = tf.equal(class_true, 1)
    mask_neg = tf.equal(class_true, 0)
    
    # localization loss (smooth L1)
    mask_pos_loc = tf.broadcast_to(mask_pos, tf.shape(loc_true))
    loss_loc = _smooth_l1_loss(tf.boolean_mask(loc_true, mask_pos_loc), 
                               tf.boolean_mask(loc_pred, mask_pos_loc))
    loss_loc = tf.reduce_mean(loss_loc)
        
    # classification loss (sigmoid crossentropy)
    focal_weight = _focal_loss(class_true, class_pred)
    loss_class = tf.keras.losses.binary_crossentropy(class_true, class_pred, from_logits=False)[..., tf.newaxis]
    loss_class = focal_weight * loss_class * (tf.cast(mask_pos, tf.float32) + tf.cast(mask_neg, tf.float32))
    loss_class = tf.reduce_mean(loss_class)
    
    return loss_loc, loss_class