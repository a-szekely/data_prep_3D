import os
import subprocess
import logging
import warnings

import scipy.ndimage
import numpy as np
from skimage import io
from skimage import util
import tensorflow as tf


class DataManager:
    def __init__(self, tfrecord_dir, mask_dir, scan_dir, swc_dir, raw_dir, tif_dir, v3d_path='/scratch/Vaa3D/vaa3d'):
        self.tfrecord_dir = tfrecord_dir.rstrip('/')
        self.mask_dir = mask_dir.rstrip('/')
        self.scan_dir = scan_dir.rstrip('/')
        self.swc_dir = swc_dir.rstrip('/')
        self.raw_dir = raw_dir.rstrip('/')
        self.tif_dir = tif_dir.rstrip('/')
        self.v3d_path = v3d_path.rstrip('/')

    def process_all_swc(self):
        files = os.listdir(self.swc_dir)
        files.remove('.gitkeep')

        for file in files:
            file_name = file.split('.')[0]
            scan = io.imread('{}/{}.scan.tifs'.format(self.scan_dir, file_name))
            shape = scan.shape
            self._swc2raw(file_name, shape)
            self._raw2tif(file_name)

    def process_all_split(self, window_shape, stride):
        filenames = os.listdir(self.tif_dir)
        filenames.remove('.gitkeep')

        masks = [filename for filename in filenames if filename.split('.')[1] == 'mask']  # oh. the. horror.
        scans = [filename for filename in filenames if filename.split('.')[1] == 'scan']

        masks.sort()
        scans.sort()

        for i in range(0, len(scans)):
            print('{} done out of {}'.format(i, len(scans)))
            statinfo = os.stat('/'.join([self.tif_dir, scans[i]]))
            if statinfo.st_size < 5.12e8:
                mask_windows = self.process_one_split(masks[i], window_shape, stride)
                scan_windows = self.process_one_split(scans[i], window_shape, stride)

                pixel_scale = np.asarray(window_shape) / np.asarray(mask_windows[0, 0, 0].shape)

                z_max = mask_windows.shape[0]
                y_max = mask_windows.shape[1]
                x_max = mask_windows.shape[2]

                for z in range(0, z_max):
                    for y in range(0, y_max):
                        for x in range(0, x_max):
                            if np.sum(mask_windows[z, y, x]) != 0:
                                n = z * y_max * x_max + y * x_max + x
                                mask_resampled = self._resample(mask_windows[z, y, x], pixel_scale)
                                scan_resampled = self._resample(scan_windows[z, y, x], pixel_scale)

                                assert mask_resampled.shape == window_shape
                                assert scan_resampled.shape == window_shape

                                animal = masks[i].split('_')[0]
                                io.imsave('{}/{}_{}_{}.scan.tifs'.format(self.scan_dir, animal,
                                                                        str(i).zfill(2),
                                                                        str(n).zfill(6)),
                                          scan_resampled)
                                io.imsave('{}/{}_{}_{}.mask.tifs'.format(self.mask_dir, animal,
                                                                        str(i).zfill(2),
                                                                        str(n).zfill(6)),
                                          mask_resampled)

                                print('Saved mask and scan of {}, with shape {}x{}x{} and sum {}'.format(animal,
                                                                                                         window_shape[
                                                                                                             0],
                                                                                                         window_shape[
                                                                                                             1],
                                                                                                         window_shape[
                                                                                                             2],
                                                                                                         np.sum(
                                                                                                             mask_resampled)))

    def process_all_tfrecord(self):
        masks = os.listdir(self.mask_dir)
        scans = os.listdir(self.scan_dir)
        masks.remove('.gitkeep')
        scans.remove('.gitkeep')

        masks.sort()
        scans.sort()

        assert len(scans) == len(masks)

        for i in range(0, len(masks)):
            mask_stack = io.imread('/'.join([self.mask_dir, masks[i]]))
            scan_stack = io.imread('/'.join([self.scan_dir, scans[i]]))

            assert mask_stack.shape == scan_stack.shape

            mask_feature = self._ndarray2feature(mask_stack)
            scan_feature = self._ndarray2feature(scan_stack)

            example = self._features2serial(scan_feature, mask_feature)

            path_tfrecord = '{}/{}.tfrecord'.format(self.tfrecord_dir, scans[i].split('.')[0])
            with tf.python_io.TFRecordWriter(path_tfrecord) as writer:
                writer.write(example)

            print('{} done out of {}'.format((i + 1), len(masks)))

    def process_one_split(self, filename, window_shape, stride):
        pixel_dims = self._get_pixel_dims(filename)
        pixel_dim_scales = pixel_dims / np.min(pixel_dims)

        window_shape_scaled = np.asarray(np.floor(window_shape / pixel_dim_scales))
        window_shape_scaled = window_shape_scaled.astype(dtype=np.int)

        stride_scaled = np.asarray(np.floor(stride / pixel_dim_scales))
        stride_scaled = stride_scaled.astype(dtype=np.int)

        stack = io.imread('/'.join([self.tif_dir, filename]))
        stack_shape = np.asarray(stack.shape)

        padded_shape = stack_shape + window_shape_scaled - np.mod(stack_shape, stride_scaled)
        stack_padded = self._pad(stack, padded_shape)
        windows = self._split(stack_padded, window_shape_scaled, stride_scaled)

        return windows

    def _swc2raw(self, file_name, shape):
        assert len(shape) == 3

        in_path = '{}/{}.swc'.format(self.swc_dir.rstrip('/'), file_name)
        out_path = '{}/{}.mask.raw'.format(self.raw_dir.rstrip('/'), file_name)

        logging.info("Converting file at{} to raw format.".format(in_path))

        cmd_swc2raw = '-x swc_to_maskimage_cylinder_unit -f swc2mask'
        cmd_size = '-p {} {} {}'.format(shape[0], shape[1], shape[2])

        cmd_full = ' '.join([self.v3d_path, cmd_swc2raw, cmd_size, '-i', in_path, '-o', out_path])

        return subprocess.call(cmd_full, shell=True)

    def _raw2tif(self, file_name):
        cmd_convert = '-x libconvert_file_format -f convert_format'

        in_path = '{}/{}.mask.raw'.format(self.raw_dir.rstrip('/'), file_name)
        out_path = '{}/{}.mask.tifs'.format(self.tif_dir.rstrip('/'), file_name)

        cmd_full = ' '.join([self.v3d_path, cmd_convert, '-i', in_path, '-o', out_path])

        return subprocess.call(cmd_full, shell=True)

    @staticmethod
    def _get_pixel_dims(filename):
        pixel_dim_x = filename.split('.')[0].split('_')[2]
        pixel_dim_y = filename.split('.')[0].split('_')[3]
        pixel_dim_z = filename.split('.')[0].split('_')[4]

        pixel_dims = np.array([pixel_dim_z.replace('-', '.'),
                               pixel_dim_y.replace('-', '.'),
                               pixel_dim_x.replace('-', '.')]).astype('float32')

        return pixel_dims

    @staticmethod
    def _pad(stack, padded_shape):
        assert len(padded_shape) == 3

        padded_stack = np.zeros(padded_shape)
        padded_stack[0:stack.shape[0], 0:stack.shape[1], 0:stack.shape[2]] = stack

        assert all(padded_stack.shape == padded_shape)

        return padded_stack

    @staticmethod
    def _resample(stack, pixel_dims):
        pixel_scale = pixel_dims / min(pixel_dims)
        scaled_stack = scipy.ndimage.zoom(stack, pixel_scale)

        return np.ascontiguousarray(scaled_stack, 'int32')

    @staticmethod
    def _split(stack, window_shape, stride):
        assert len(window_shape) == 3
        assert len(stride) == 3
        for i in range(0, 3):
            residual = (stack.shape[i] - window_shape[i]) % stride[i]
            assert residual == 0

        samples = util.view_as_windows(stack, window_shape, stride)

        return samples

    @staticmethod
    def _ndarray2feature(stack):
        stack_tensor = tf.convert_to_tensor(stack)
        stack_bytestring = tf.serialize_tensor(stack_tensor).numpy()
        stack_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[stack_bytestring]))

        return stack_feature

    @staticmethod
    def _features2serial(input_feature, target_feature):
        features = tf.train.Features(feature={'input': input_feature,
                                              'target': target_feature})
        example = tf.train.Example(features=features)
        example_serial = example.SerializeToString()

        return example_serial


# TODO tests

if __name__ == '__main__':
    tf.enable_eager_execution()
    logging.basicConfig(level=logging.DEBUG)
    warnings.filterwarnings('ignore')

    TFRECORD_DIR = '/var/home/4thyr.oct2018/as2554/neuroseg/data/gold166/tfrecords/'
    MASK_DIR = '/var/home/4thyr.oct2018/as2554/neuroseg/data/gold166/masks/'
    SCAN_DIR = '/var/home/4thyr.oct2018/as2554/neuroseg/data/gold166/scans/'
    SWC_DIR = '/var/home/4thyr.oct2018/as2554/neuroseg/data/gold166/original/swc/'
    TIF_DIR = '/var/home/4thyr.oct2018/as2554/neuroseg/data/gold166/original/tifs/'
    RAW_DIR = '/var/home/4thyr.oct2018/as2554/neuroseg/data/gold166/original/raw/'

    converter = DataManager(TFRECORD_DIR, MASK_DIR, SCAN_DIR, SWC_DIR, RAW_DIR, TIF_DIR)
    # converter.process_all_split((128, 128, 128), (128, 128, 128))
    converter.process_all_tfrecord()
