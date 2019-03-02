import logging
import tensorflow as tf
from dataprep import tiftools, tfrecordtools

tf.enable_eager_execution()
logging.basicConfig(level=logging.DEBUG)

#some comment

#tiftools.split_all((128, 128, 128), (128, 128, 128))
tfrecordtools.process_all_tfrecord()
