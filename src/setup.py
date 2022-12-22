# Libraries
import os
import shutil
from pathlib import Path
import requests
import zipfile
import random
import logging as log
from tqdm import tqdm

# Main method
def setup_folder_structure_FTYM(
    pathdir: str='input'
):
    """Prepare folder structurte with image dataset.

    Args:
        path (str, optional): Folder where dataset will be unfolded. Defaults to './input'.
    """

    if not os.path.isdir(os.path.join(pathdir, 'train', 'images')):
        #
        # PART I: Training data
        #

        # Create directory and change working one
        Path(os.path.join(pathdir, 'train', 'images')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(pathdir, 'train', 'annotations')).mkdir(parents=True, exist_ok=True)
        # Fetch information and load
        log.debug("Fetch dataset")
        with requests.get("https://facesyntheticspubwedata.blob.core.windows.net/iccv-2021/dataset_1000.zip", allow_redirects=True) as r:
            open(os.path.join(pathdir, 'train', 'FTYM.zip'), 'wb').write(r.content)
        # Unzip file
        with zipfile.ZipFile(os.path.join(pathdir, 'train', 'FTYM.zip'), 'r') as zipObj:
            zipObj.extractall(pathdir)
        # Remove .zip
        os.remove(os.path.join(pathdir, 'train', 'FTYM.zip'))
        # Split indices
        log.debug("Split indices")
        train_idx = random.sample(range(1000), 800)
        val_idx = [x for x in range(1000) if x not in train_idx]
        # Create folders and accomodate selected images-annotations
        log.debug("Training data")
        for idx in tqdm(train_idx):
            # Format idx as it is written in filenames
            filename_image=str(idx).zfill(6)+'.png'
            filename_mask=str(idx).zfill(6)+'_seg.png'
            filename_anot=str(idx).zfill(6)+'_ldmks.txt'
            # Move image
            os.replace(os.path.join(pathdir, filename_image), os.path.join(pathdir, 'train', 'images', filename_image))
            # Move segmentation mask
            os.replace(os.path.join(pathdir, filename_mask), os.path.join(pathdir, 'train', 'annotations', filename_mask))
            # Drop landmark annotation
            os.remove(os.path.join(pathdir, filename_anot))


        #
        # PART II: Validation data
        #

        # Create directory and change working one
        Path(os.path.join(pathdir, 'val', 'images')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(pathdir, 'val', 'annotations')).mkdir(parents=True, exist_ok=True)
        # Create folders and accomodate selected images-annotations
        log.debug("Validation data")
        for idx in tqdm(val_idx):
            # Format idx as it is written in filenames
            filename_image=str(idx).zfill(6)+'.png'
            filename_mask=str(idx).zfill(6)+'_seg.png'
            filename_anot=str(idx).zfill(6)+'_ldmks.txt'
            # Move image
            os.replace(os.path.join(pathdir, filename_image), os.path.join(pathdir, 'val', 'images', filename_image))
            # Move segmentation mask
            os.replace(os.path.join(pathdir, filename_mask), os.path.join(pathdir, 'val', 'annotations', filename_mask))
            # Drop landmark annotation
            os.remove(os.path.join(pathdir, filename_anot))
    else:
        train_idx = [int(x.split('.')[0].lstrip('0')) if x!='000000.png' else 0 for x in os.listdir(os.path.join(pathdir, 'train', 'images'))]
        val_idx = [int(x.split('.')[0].lstrip('0')) if x!='000000.png' else 0 for x in os.listdir(os.path.join(pathdir, 'val', 'images'))]
    
    #
    # Part III: Labels mapping
    #

    tag2idx = {
        'BACKGROUND' : 0,
        'SKIN' : 1,
        'NOSE' : 2,
        'RIGHT_EYE' : 3,
        'LEFT_EYE' : 4,
        'RIGHT_BROW' : 5,
        'LEFT_BROW' : 6,
        'RIGHT_EAR' : 7,
        'LEFT_EAR' : 8,
        'MOUTH_INTERIOR' : 9,
        'TOP_LIP' : 10,
        'BOTTOM_LIP' : 11,
        'NECK' : 12,
        'HAIR' : 13,
        'BEARD' : 14,
        'CLOTHING' : 15,
        'GLASSES' : 16,
        'HEADWEAR' : 17,
        'FACEWEAR' : 18,
        'IGNORE' : 255
    }

    return train_idx, val_idx, tag2idx