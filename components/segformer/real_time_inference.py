#Import necessary libraries
import gradio as gr
import os
import cv2
import numpy as np
import matplotlib
import torch
from torchvision.transforms import ToTensor
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# Auxiliary method to prepare 
def load_checkpoint(device):
    # Load checkpoint
    ckpt = torch.load('./output/best-checkpoint.bin', map_location='cpu')
    # Create model
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
            'FACEWEAR' : 18
        }
    idx2tag = {v:k for k,v in tag2idx.items()}
    feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/mit-b0',
        ignore_mismatched_sizes=True,
        num_labels=len(tag2idx),
        id2label=idx2tag,
        label2id=tag2idx,
        reshape_last_stage=True
    )
    # Load weights
    model.load_state_dict(ckpt.get('model_state_dict'))
    model.to(device)
    return idx2tag, feature_extractor, model

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
idx2tag, feature_extractor, model = load_checkpoint(device)

# Auxiliary method to generate output
def inference(image, model=model, feature_extractor=feature_extractor, device=device):
    frame = ToTensor()(image)
    with torch.no_grad():
        im_prep = feature_extractor(frame, return_tensors='pt')
        output = model(pixel_values = im_prep.get('pixel_values').to(device))
        output = torch.nn.functional.interpolate(output.get('logits').detach(), size=image.shape[:2], mode="bilinear", align_corners=False).argmax(dim=1).numpy()[0]
        my_cm = matplotlib.cm.get_cmap('jet')
        mapped_data = my_cm(output/18, bytes=False)
        return mapped_data


demo = gr.Interface(
    inference,
    gr.Image(shape=(512,512), source="webcam", streaming=True, flip=True),
    "image",
    live=True
)
demo.launch(share=True)