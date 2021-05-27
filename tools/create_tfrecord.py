import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from absl import app, flags, logging
import tensorflow as tf

logging.set_verbosity(logging.DEBUG)
logging.info('TF Version:', tf.__version__)

flags.DEFINE_string('mode', 'train', 'train or valid')
flags.DEFINE_string('dataset_path', './tmp', 'path of dataset')
flags.DEFINE_string('tfrecord_path', './widerface_train.tfrecord', 'path of output tfrecord')
FLAGS = flags.FLAGS

def checkArgvPaths():
    """Check paths from argv"""

    if not os.path.isdir(FLAGS.dataset_path):
        logging.info('Please define valid dataset path.')
        exit()
        
    if os.path.exists(FLAGS.tfrecord_path):
        logging.info(f'{FLAGS.tfrecord_path} already exists. Exit ...')
        exit()

def loader(dataset_path, mode='valid'):
    """Convert txt to infos"""
    
    if mode == 'valid':
        txt_path    = os.path.join(dataset_path, 'wider_face_split/wider_face_val_bbx_gt.txt')
        images_path = os.path.join(dataset_path, 'WIDER_val/images')
    elif mode == 'train':
        txt_path    = os.path.join(dataset_path, 'wider_face_split/wider_face_train_bbx_gt.txt')
        images_path = os.path.join(dataset_path, 'WIDER_train/images')
    else:
        logging.info('Please chose the right mode. [train/valid] Exit ...')
        exit()

    infos = []

    with open(txt_path, 'r') as f:
        isPath = True
        counter = 0

        line = f.readline()
        while line:
            line = line.strip()
            logging.debug(f'len = {len(infos)}, line : {line}')

            if isPath:
                info = {}
                info['name'] = line
                info['path'] = os.path.join(images_path, line)
                info['bboxes'] = []
                isPath = False
            else:
                if line == '0 0 0 0 0 0 0 0 0 0': # invalid infomation
                    isPath = True
                    counter = 0
                elif counter == 0:
                    counter = int(line) 
                else:
                    x1, y1, w, h = line.split(' ')[:4] # x1, y1, w, h
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = x1 + int(w)
                    y2 = y1 + int(h)
                    info['bboxes'].append([x1, y1, x2, y2])
                    counter -= 1
                    if counter == 0:
                        isPath = True
                        infos.append(info)
                        logging.debug(info)

            line = f.readline()

    return infos

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(info):
    """Creates a tf.Example message ready to be written to a file."""
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    with open(info['path'], 'rb') as f:
        img_bytes = f.read()
    name_bytes = str.encode(info['name'])
    bboxes_bytes = np.array(info['bboxes'], np.float32).tobytes()
    
    feature = {
        'image/name':   _bytes_feature(name_bytes),
        'image/encode': _bytes_feature(img_bytes),
        'label/bboxes': _bytes_feature(bboxes_bytes),
    }
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def createTFRecord(infos, tfrecord_path):
    """Writing infos to TFRecord file"""
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for info in tqdm(infos):
            example = serialize_example(info)
            writer.write(example)

def main(argv):
    
    checkArgvPaths()
    
    # get infos from dataset ...
    logging.info(f'Loading ... {FLAGS.dataset_path}')
    infos = loader(dataset_path=FLAGS.dataset_path, mode=FLAGS.mode)

    
    # write sample to tfrecord file ...
    logging.info(f'Writing {len(infos)} samples to TFRecord file ...')
    createTFRecord(infos, FLAGS.tfrecord_path)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass