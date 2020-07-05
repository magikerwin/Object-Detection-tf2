from absl import app, flags, logging
flags.DEFINE_string('cfg_path', './cfg/toy.yaml', 'path of config yaml')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
FLAGS = flags.FLAGS

import os
from modules.utils import load_yaml, load_dataset

import tensorflow as tf
logging.info('TF Version:', tf.__version__)

def main(argv):

    # init
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    
    cfg = load_yaml(FLAGS.cfg_path)

    # load dataset
    train_dataset = load_dataset(cfg)

    for x, y in train_dataset:
        break
    print('x.shape:', x.shape)
    print('y:', y)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass