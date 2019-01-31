import os
import logging
import warnings
import re

import numpy as np
import scipy.ndimage
from skimage import io
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
        try:
            split_one(scans[i], masks[i], window_shape, stride)
            logging.info("{} out of {} done".format((i + 1), len(scans)))
        except MemoryError as e:
            logging.exception("The splitting process ran out of memory.", e)


def split_one(scan_name, mask_name, window, stride):
    """Split and resample one scan and mask pair, and save the results in the tif directory.

    Split one mask and scan pair according to window_shape and stride. Then resample it to recover isotropic voxels.
    Only resample and save the blocks which contain neuron segments.
    """
    datapoint_name = "{}_{}".format(scan_name.split('_')[0], scan_name.split('_')[1])

    # Sample the scan and mask with a window transformed such that it has the same 3D resolution as the scan
    window_scaled, stride_scaled = get_scaled_sampler(scan_name, window, stride)

    # Read scan and mask stacks
    mask = io.imread('/'.join([dataprep.constants.TIF_DIR, mask_name]))
    scan = io.imread('/'.join([dataprep.constants.TIF_DIR, scan_name]))

    mask_shape = np.asarray(mask.shape)
    scan_shape = np.asarray(scan.shape)

    # Ensure that the inputs are compatible
    assert all(mask_shape == scan_shape)

    # Pad mask and scan with zeros so that an integer number of samples can exactly be taken from them
    padded_shape = mask_shape + (window_scaled - np.mod(mask_shape, stride_scaled))
    mask = pad(mask, padded_shape)
    scan = pad(scan, padded_shape)

    # Move the window over the mask and scan pair, and resample and save each sample pair containing part of the neuron
    moves_along_axis = ((padded_shape - window_scaled) / stride_scaled) + 1
    moves_along_axis = np.ndarray.astype(moves_along_axis, dtype=int)
    for k in range(0, moves_along_axis[0]):
        for j in range(0, moves_along_axis[1]):
            for i in range(0, moves_along_axis[2]):
                mask_sample = mask[(k * window_scaled[0]):((k + 1) * window_scaled[0]),
                                   (j * window_scaled[1]):((j + 1) * window_scaled[1]),
                                   (i * window_scaled[2]):((i + 1) * window_scaled[2])]

                # Check the mask sample for the presence of neuron segment
                if np.sum(mask_sample) > 0:
                    scan_sample = scan[(k * window_scaled[0]):((k + 1) * window_scaled[0]),
                                       (j * window_scaled[1]):((j + 1) * window_scaled[1]),
                                       (i * window_scaled[2]):((i + 1) * window_scaled[2])]

                    # Resample mask and scan to ensure isotropic voxels
                    mask_sample = resample(mask_sample, window)
                    mask_sample = mask_sample.astype(dtype=np.int32)

                    scan_sample = resample(scan_sample, window)
                    scan_sample = scan_sample.astype(dtype=np.int32)

                    # Disable warnings while saving samples, which arise from the low contrast nature of the mask
                    warnings.filterwarnings('ignore')

                    n = str(k * moves_along_axis[1] * moves_along_axis[2] + j * moves_along_axis[2] + i).zfill(6)
                    io.imsave("{}/{}_{}.mask.tif".format(dataprep.constants.MASK_DIR, datapoint_name, n), mask_sample)
                    io.imsave("{}/{}_{}.scan.tif".format(dataprep.constants.SCAN_DIR, datapoint_name, n), scan_sample)

                    logging.debug("{}_{} was successfully saved".format(datapoint_name, n))

                    warnings.filterwarnings('default')

    logging.info('Successfully split and resampled {}'.format(datapoint_name))


def get_scaled_sampler(filename, window_shape, stride):
    # extract pixel dimensions from filename
    pixel_dims = get_pixel_dims(filename)

    # calculate how much the window and stride need to be scaled in each dimension
    pixel_dim_scales = np.min(pixel_dims) / pixel_dims

    # rescale window and stride
    window_scaled = np.asarray(np.floor(window_shape * pixel_dim_scales), dtype=np.int)
    stride_scaled = np.asarray(np.floor(stride * pixel_dim_scales), dtype=np.int)

    logging.debug("Window shape has been modified to {} and stride to {} to align with voxel resolution of {}".format(
        window_scaled, stride_scaled, pixel_dims))

    return window_scaled, stride_scaled


def get_pixel_dims(filename):
    pixel_dim_x = filename.split('.')[0].split('_')[2]
    pixel_dim_y = filename.split('.')[0].split('_')[3]
    pixel_dim_z = filename.split('.')[0].split('_')[4]

    pixel_dims = np.array([pixel_dim_z.replace('-', '.'),
                           pixel_dim_y.replace('-', '.'),
                           pixel_dim_x.replace('-', '.')]).astype('float32')
    return pixel_dims


def pad(stack, padded_shape):
    padded_stack = np.zeros(padded_shape)
    padded_stack[0:stack.shape[0], 0:stack.shape[1], 0:stack.shape[2]] = stack

    return padded_stack


def resample(stack, output_shape):
    input_shape = np.asarray(stack.shape)
    output_shape = np.asarray(output_shape)

    scale = output_shape / input_shape

    # Resample stack using cubic interpolation
    scaled_stack = scipy.ndimage.zoom(stack, scale, order=1)

    # NB. Fiji only likes int32
    return np.ascontiguousarray(scaled_stack, 'int32')
