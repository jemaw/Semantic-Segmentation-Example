"""
Downloads extracts and prepares data for semantic segmentation
Uses data from Kaggle Data Science Bowl 2018:
    https://data.broadinstitute.org/bbbc/BBBC038/
"""
# import argparse
import os
import sys

import requests
import numpy as np
import zipfile
import glob
from skimage import io


DATA_URL = "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip"
IMAGE_FOLDER = 'images'
MASK_FOLDER = 'masks'





def download_file(url, local_filename=None):
    """See: https://stackoverflow.com/questions/16694907/how-to-download-large-file-in-python-with-requests-py"""
    if local_filename is None:
        local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    total_length = int(r.headers.get('content-length'))



    with open(local_filename, 'wb') as f:
        progress = 0
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                progress += len(chunk)
                done = int(50*progress/total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                sys.stdout.flush()
    return local_filename

def combine_masks(mask_paths):
    """Combines the masks for the individual instances into
    one single mask for the entire image

    Args:
        mask_paths: list containing paths to all masks of one image
    """
    tmp_im = io.imread(mask_paths[0])
    total_mask = np.zeros(tmp_im.shape, tmp_im.dtype)
    for mask_path in mask_paths:
        im = io.imread(mask_path)
        total_mask += im

    assert np.max(total_mask) <= 255
    return total_mask.astype(np.uint8)


def move_data(data_folder, image_folder, mask_folder):
    image_out_folder = image_folder
    mask_out_folder = mask_folder
    os.makedirs(image_out_folder, exist_ok=True)
    os.makedirs(mask_out_folder, exist_ok=True)

    folder_glob_pattern = os.path.join(data_folder, '*/')
    folders = glob.glob(folder_glob_pattern)

    for i, folder in enumerate(folders):
        idx = os.path.basename(os.path.dirname(folder))

        image_glob_pattern = os.path.join(folder, image_folder, '*.png')
        images = glob.glob(image_glob_pattern)
        assert len(images) == 1
        image_path = images[0]
        image = io.imread(image_path)

        mask_glob_pattern = os.path.join(folder, mask_folder, '*.png')
        masks = glob.glob(mask_glob_pattern)
        assert len(masks) > 0
        total_mask = combine_masks(masks)

        image_out_path = os.path.join(image_out_folder, idx + '.png')
        mask_out_path = os.path.join(mask_out_folder, idx + '_mask.png')

        io.imsave(image_out_path, image)
        io.imsave(mask_out_path, total_mask)


        done = int(50*i/len(folders))
        sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)))
        sys.stdout.flush()

def main():
    local_filename = DATA_URL.split('/')[-1]
    if os.path.isfile(local_filename):
        print('Skipping download, file %s already exists' % (local_filename))
    else:
        print('Downloading data')
        download_file(DATA_URL, local_filename)
        print('Finished Downloading')

    tmpdata_folder = 'tmpdata'
    if os.path.isdir(tmpdata_folder):
        print('Skipping extraction, folder %s already exists' % (tmpdata_folder))
    else:
        with zipfile.ZipFile(local_filename, 'r') as zip_ref:
            print('Unzipping data')
            zip_ref.extractall(tmpdata_folder)

    print('Preparing and moving data')
    move_data(tmpdata_folder, IMAGE_FOLDER, MASK_FOLDER)



if __name__ == "__main__":
    main()
