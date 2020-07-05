import tensorflow as tf

def _parse_tfrecord(img_dim):
    def parse_tfrecord(tfrecord):
        feature = {
            'image/name':   tf.io.FixedLenFeature([], tf.string), #_bytes_feature(name_str),
            'image/encode': tf.io.FixedLenFeature([], tf.string), #_bytes_feature(img_str),
            'label/bboxes': tf.io.FixedLenFeature([], tf.string), #_bytes_feature(bboxes_str),
        }

        example = tf.io.parse_single_example(tfrecord, feature)
        img = tf.image.decode_jpeg(example['image/encode'], channels=3)
        labels = tf.io.decode_raw(example['label/bboxes'], tf.float32)
        return img, labels
    return parse_tfrecord

def load_tfrecord_dataset(path_tfrecord, img_dim=100, batch_size=1,
                          shuffle=True, buffer_size=10240):
    """load dataset from tfrecord"""
    
    raw_dataset = tf.data.TFRecordDataset(path_tfrecord)
    raw_dataset = raw_dataset.repeat()
    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)

    parser = _parse_tfrecord(img_dim)

    dataset = raw_dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset