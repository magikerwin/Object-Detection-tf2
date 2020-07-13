import tensorflow as tf

def flip_horizontal(img, bboxes):
    """flip/mirror image horizontally"""
    w_src = tf.cast(tf.shape(img)[1], tf.float32)
    
    flip_func = lambda : (
        tf.image.flip_left_right(img), 
        tf.stack([w_src - bboxes[..., 0], bboxes[..., 1],
                  w_src - bboxes[..., 2], bboxes[..., 3],], axis=-1)
    )
    flip_case = tf.random.uniform([], 0, 2, dtype=tf.int32)
    img_dst, bboxes_dst = tf.case([(tf.equal(flip_case, 0), flip_func)],
                                  default=lambda: (img, bboxes))
    
    return img_dst, bboxes_dst

def flip_vertical(img, bboxes):
    """flip/mirror image vertically"""
    h_src = tf.cast(tf.shape(img)[0], tf.float32)
    
    flip_func = lambda : (
        tf.image.flip_up_down(img), 
        tf.stack([bboxes[..., 0], h_src - bboxes[..., 1],
                  bboxes[..., 2], h_src - bboxes[..., 3],], axis=-1)
    )
    flip_case = tf.random.uniform([], 0, 2, dtype=tf.int32)
    img_dst, bboxes_dst = tf.case([(tf.equal(flip_case, 0), flip_func)],
                                  default=lambda: (img, bboxes))
    
    return img_dst, bboxes_dst