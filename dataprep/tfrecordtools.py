import os
import logging
import time

import numpy as np

import tensorflow as tf
from skimage import io
from matplotlib import pyplot as plt

import dataprep.constants


def process_all_tfrecord():

    # Ensure that Eager Execution is on. Otherwise it tends to run out of memory.
    if not tf.python.context.executing_eagerly():
        raise RuntimeError("dataprep.process_all_tfrecord is not supported without eager_execution")

    # Find all mask and scan files, and sort them (os.listdir reads files in a random order)
    mask_files = os.listdir(dataprep.constants.MASK_DIR)
    scan_files = os.listdir(dataprep.constants.SCAN_DIR)
    mask_files.sort()
    scan_files.sort()

    # Make sure there are as many masks as scans
    assert len(mask_files) == len(scan_files)

    for i in range(0, len(mask_files)):
        # Assert that the i^th mask corresponds to the i^th scan
        assert mask_files[i].split('.')[0] == scan_files[i].split('.')[0]

        mask = io.imread('/'.join([dataprep.constants.MASK_DIR, mask_files[i]]))
        scan = io.imread('/'.join([dataprep.constants.SCAN_DIR, scan_files[i]]))

        mask_feature = ndarray2feature(mask)
        scan_feature = ndarray2feature(scan)

        example = features2serial(scan_feature, mask_feature)

        path_tfrecord = '{}/{}.tfrecord'.format(dataprep.constants.TFRECORD_DIR, scan_files[i].split('.')[0])

        with tf.python_io.TFRecordWriter(path_tfrecord) as writer:
            writer.write(example)

        logging.info("Written {}.tfrecord - {} out of {} done".format(scan_files[i].split('.')[0],
                                                                      (i + 1), len(mask_files)))


def ndarray2feature(stack):
    stack_tensor = tf.convert_to_tensor(stack)
    stack_bytestring = tf.serialize_tensor(stack_tensor).numpy()
    stack_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[stack_bytestring]))

    return stack_feature


def features2serial(input_feature, target_feature):
    features = tf.train.Features(feature={'input': input_feature,
                                          'target': target_feature})
    example = tf.train.Example(features=features)
    example_serial = example.SerializeToString()

    return example_serial


def visualise_example(serialized_example):
    feature_description = {'input': tf.FixedLenFeature([], tf.string),
                           'target': tf.FixedLenFeature([], tf.string)}

    example = tf.parse_single_example(serialized_example, feature_description)

    input = tf.parse_tensor(example['input'], tf.int32)
    target = tf.parse_tensor(example['target'], tf.int32)

    input = tf.expand_dims(input, 3)
    target = tf.expand_dims(target, 3)

    input_flat = np.squeeze(tf.math.reduce_sum(input, axis=0))
    target_flat = np.squeeze(tf.math.reduce_sum(target, axis=0))

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(input_flat)
    fig.add_subplot(1, 2, 2)
    plt.imshow(target_flat)
    plt.show()
    time.sleep(5)
    plt.close()




