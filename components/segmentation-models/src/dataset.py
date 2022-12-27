# Libraries
import os
import cv2
import numpy as np
import torch
import albumentations as A
from segmentation_models_pytorch.encoders import get_preprocessing_params
from torch.utils.data import Dataset

# Main class
class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, path, idx_list, encoder_flag:str='timm-mobilenetv3_large_100', encoder_weights:str='imagenet', img_size:int=128, num_labels:int=19, is_train:bool=True):
        """
        Object that fetches images and masks.

        Args:
            root_dir (str): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            is_train (bool): Whether to apply training augmentation or validation
        """
        super(ImageSegmentationDataset, self).__init__()
        self.path = path
        self.idx_list = idx_list
        self.prep_params = get_preprocessing_params(encoder_flag, pretrained=encoder_weights)
        self.img_size = img_size
        self.num_labels = num_labels
        self.is_train = is_train

        self.img_dir = os.path.join(self.path, "images")
        self.ann_dir = os.path.join(self.path, "annotations")
        self.num_samples = len(self.idx_list)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx = self.idx_list[idx]
        im, annot=str(idx).zfill(6)+'.png', str(idx).zfill(6)+'_seg.png'
        image = cv2.imread(os.path.join(self.img_dir, im))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        segmentation_map = cv2.imread(os.path.join(self.ann_dir, annot))
        segmentation_map = cv2.cvtColor(segmentation_map, cv2.COLOR_BGR2GRAY)

        if self.is_train:
            pipe = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size, interpolation=cv2.INTER_LINEAR, always_apply=True),
                A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),       
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=.75, shift_limit=0.5, p=1)                  
                    ], p=0.8),
                A.CLAHE(clip_limit=2, tile_grid_size=(16,16), p=0.8),
                A.RandomBrightnessContrast(brightness_limit=(-.1,.1), contrast_limit=(-.1,.1), p=0.8),    
                A.RandomGamma(p=0.8)])
            augmented = pipe(image=image, mask=segmentation_map)
            encoded_inputs = self.preprocess_input(augmented['image'], **self.prep_params)
        else:
            pipe = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size, interpolation=cv2.INTER_LINEAR, always_apply=True),
                A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, always_apply=True, border_mode=cv2.BORDER_CONSTANT)
                ])
            augmented = pipe(image=image, mask=segmentation_map)
            img = self.preprocess_input(augmented['image'], **self.prep_params)
        return (torch.from_numpy(img).squeeze(), torch.from_numpy(augmented['mask']).squeeze())

    @staticmethod
    def preprocess_input(x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs):

            if input_space == "BGR":
                x = x[..., ::-1].copy()

            if input_range is not None:
                if x.max() > 1 and input_range[1] == 1:
                    x = x / 255.0

            if mean is not None:
                mean = np.array(mean)
                x = x - mean

            if std is not None:
                std = np.array(std)
                x = x / std

            return np.transpose(x, (2,0,1))