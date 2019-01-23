import os
import logging
import warnings
import re

import numpy as np
import scipy.ndimage  # TODO do I actually need this?
from skimage import io
from skimage import util

import dataprep.constants


def split_all(window_shape, stride):
    """Split and resample all scans and masks in the tif directory.

    Resample all masks and scans to recover isotropic voxels. Then, split them into identical blocks whose size is
    determined by window_shape and the overlap is determined by stride (overlap is window_shape-stride).
    """

    # Separate mask and scan files from the tif directory
    tif_files = os.listdir(dataprep.constants.TIF_DIR)

    r_mask = re.compile(".*mask.*")
    r_scan = re.compile(".*scan.*")

    masks = list(filter(r_mask.match, tif_files))
    scans = list(filter(r_scan.match, tif_files))

    logging.info("{} masks, and {} scans were found in the tif directory".format(len(masks), len(scans)))

    # os.listdir reads filenames in a random order
    masks.sort()
    scans.sort()

    # Split each scan-mask pair
    for i in range(0, len(scans)):
        datapoint_name = "{}_{}".format(scans[i].split('_')[0], scans[i].split('_')[1])

        # Large files cause Out of Memory errors, so avoid splitting them
        # TODO Change splitting algorithm to index into the scan/mask ndarrays instead of using util.view_as_windows

        scan_stats = os.stat('/'.join([dataprep.constants.TIF_DIR, scans[i]]))
        if scan_stats.st_size > 5.12e8:
            # Skip files that are larger than ~0.5 Gigabyte
            logging.warning("{} is too large, and will be ignored".format(datapoint_name))
            continue

        # Split scan and mask
        mask_windows = split_one(masks[i], window_shape, stride)
        scan_windows = split_one(scans[i], window_shape, stride)

        pixel_scale = np.asarray(window_shape) / np.asarray(mask_windows[0, 0, 0].shape)

        z_max = mask_windows.shape[0]
        y_max = mask_windows.shape[1]
        x_max = mask_windows.shape[2]

        for z in range(0, z_max):
            for y in range(0, y_max):
                for x in range(0, x_max):
                    if np.sum(mask_windows[z, y, x]) != 0:
                        id = str(z * y_max * x_max + y * x_max + x).zfill(6)

                        # Resample the window so it has the desired shape
                        mask_resampled = resample(mask_windows[z, y, x], pixel_scale)
                        scan_resampled = resample(scan_windows[z, y, x], pixel_scale)

                        # Save mask and scan segments
                        io.imsave('{}/{}_{}.scan.tifs'.format(dataprep.constants.SCAN_DIR, datapoint_name, id),
                                  scan_resampled)
                        io.imsave('{}/{}_{}.mask.tifs'.format(dataprep.constants.MASK_DIR, datapoint_name, id),
                                  mask_resampled)

                        logging.debug("Saved mask and scan of {}_{}.".format(datapoint_name, id))

        logging.info("Successfully split {} - {} out of {} done".format(datapoint_name, (i+1), len(scans)))


def split_one(filename, window_shape, stride):
    pixel_dims = get_pixel_dims(filename)
    pixel_dim_scales = pixel_dims / np.min(pixel_dims)

    window_shape_scaled = np.asarray(np.floor(window_shape / pixel_dim_scales))
    window_shape_scaled = window_shape_scaled.astype(dtype=np.int)

    stride_scaled = np.asarray(np.floor(stride / pixel_dim_scales))
    stride_scaled = stride_scaled.astype(dtype=np.int)

    stack = io.imread('/'.join([dataprep.constants.TIF_DIR, filename]))
    stack_shape = np.asarray(stack.shape)

    padded_shape = stack_shape + window_shape_scaled - np.mod(stack_shape, stride_scaled)
    stack_padded = pad(stack, padded_shape)
    windows = split(stack_padded, window_shape_scaled, stride_scaled)

    return windows


def get_pixel_dims(filename):
    pixel_dim_x = filename.split('.')[0].split('_')[2]
    pixel_dim_y = filename.split('.')[0].split('_')[3]
    pixel_dim_z = filename.split('.')[0].split('_')[4]

    pixel_dims = np.array([pixel_dim_z.replace('-', '.'),
                           pixel_dim_y.replace('-', '.'),
                           pixel_dim_x.replace('-', '.')]).astype('float32')
    return pixel_dims


def pad(stack, padded_shape):
    assert len(padded_shape) == 3

    padded_stack = np.zeros(padded_shape)
    padded_stack[0:stack.shape[0], 0:stack.shape[1], 0:stack.shape[2]] = stack

    assert all(padded_stack.shape == padded_shape)

    return padded_stack


def resample(stack, pixel_dims):
    pixel_scale = pixel_dims / min(pixel_dims)
    scaled_stack = scipy.ndimage.zoom(stack, pixel_scale)

    return np.ascontiguousarray(scaled_stack, 'int32')


def split(stack, window_shape, stride):
    assert len(window_shape) == 3
    assert len(stride) == 3
    for i in range(0, 3):
        residual = (stack.shape[i] - window_shape[i]) % stride[i]
        assert residual == 0

    samples = util.view_as_windows(stack, window_shape, stride)
    return samples
