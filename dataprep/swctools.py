import os
import subprocess
import logging

from skimage import io

import dataprep.constants


def process_all_swc():
    """Convert and move all .swc mask files - found in the directory prescribed by constants.py - to .tif files."""

    # List all swc files
    swc_files = os.listdir(dataprep.constants.SWC_DIR)

    for file in swc_files:
        # Extract scan name from .swc filename
        scan_name = file.split('.')[0]

        # Load the appropriate .tif scan file to figure out its shape (required parameter by Vaa3D
        scan = io.imread('{}/{}.scan.tifs'.format(dataprep.constants.TIF_DIR, scan_name))
        shape = scan.shape
        swc2raw(scan_name, shape)
        raw2tif(scan_name)


def swc2raw(file_name, shape):
    """Conver the named .swc file to .raw format using Vaa3D."""

    logging.info("Converting {} to [.raw] format.".format(file_name))

    # Compose Vaa3D CLI command
    input_path = '{}/{}.swc'.format(dataprep.constants.SWC_DIR, file_name)
    output_path = '{}/{}.mask.raw'.format(dataprep.constants.RAW_DIR, file_name)

    cmd_swc2raw = '-x swc_to_maskimage_cylinder_unit -f swc2mask'
    cmd_size = '-p {} {} {}'.format(shape[0], shape[1], shape[2])

    cmd_full = ' '.join([dataprep.constants.V3D_PATH, cmd_swc2raw, cmd_size, '-i', input_path, '-o', output_path])

    return subprocess.call(cmd_full, shell=True)


def raw2tif(file_name):
    """Convert the named .raw file to .tif format using Vaa3D """

    logging.info("Converting {} to [.tif] format".format(file_name))

    # Compose Vaa3D CLI command
    in_path = '{}/{}.mask.raw'.format(dataprep.constants.RAW_DIR, file_name)
    out_path = '{}/{}.mask.tifs'.format(dataprep.constants.TIF_DIR, file_name)

    cmd_convert = '-x libconvert_file_format -f convert_format'

    cmd_full = ' '.join([dataprep.constants.V3D_PATH, cmd_convert, '-i', in_path, '-o', out_path])

    return subprocess.call(cmd_full, shell=True)
