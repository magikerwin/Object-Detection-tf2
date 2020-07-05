import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from absl import app, flags, logging
import tensorflow as tf

logging.set_verbosity(logging.DEBUG)
logging.info('TF Version:', tf.__version__)

# flags.DEFINE_string('dataset_path', './data/widerface/train', 'path of dataset')
flags.DEFINE_string('dataset_path', 'C:\\Users\\qwerz\\working\\2--Dataset\\WiderFace\\drive-download-20190907T020223Z-001', 'path of dataset')
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
        txt_path    = os.path.join(dataset_path, 'wider_face_val_bbx_gt_toy.txt')
        images_path = os.path.join(dataset_path, 'WIDER_val\images')
    elif mode == 'train':
        txt_path    = os.path.join(dataset_path, 'wider_face_train_bbx_gt.txt')
        images_path = os.path.join(dataset_path, 'WIDER_train\images')
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
                if counter == 0:
                    counter = int(line) 
                else:
                    bboxes = line.split(' ')[:4] # x,y,h,w
                    info['bboxes'].append(bboxes)
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
        img_str = f.read()
    name_str = str.encode(info['name'])
    bboxes_str = np.array(info['bboxes'], np.float32).tostring()

    feature = {
        'image/name':   _bytes_feature(name_str),
        'image/encode': _bytes_feature(img_str),
        'label/bboxes': _bytes_feature(bboxes_str),
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
    infos = loader(FLAGS.dataset_path, mode='valid')

    
    # write sample to tfrecord file ...
    logging.info(f'Writing {len(infos)} samples to TFRecord file ...')
    createTFRecord(infos, FLAGS.tfrecord_path)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass