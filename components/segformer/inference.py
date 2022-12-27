#Import necessary libraries
from flask import Flask, render_template, Response
import os
import cv2
import numpy as np
import matplotlib
import torch
from torchvision.transforms import ToTensor
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# Auxiliary method to prepare 
def load_checkpoint():
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
    # Use TorchScript inference (oneDNN Graph)
    torch.jit.enable_onednn_fusion(True)
    # sample input should be of the same shape as expected inputs
    sample_input = feature_extractor(torch.rand(3, 512, 512), return_tensors='pt').get('pixel_values')
    # Tracing the model with example input
    traced_model = torch.jit.trace(model, sample_input, strict=False)
    # Invoking torch.jit.freeze
    traced_model = torch.jit.freeze(traced_model)
    # Warmup inference
    with torch.no_grad():
        # a couple of warmup runs
        for _ in range(5):
            _ = traced_model(sample_input)
    return idx2tag, feature_extractor, traced_model

# Auxiliary method to generate output
def inference(model, feature_extractor, image):
    with torch.no_grad():
        im_prep = feature_extractor(image, return_tensors='pt')
        output = model(pixel_values = im_prep.get('pixel_values'))
        output = torch.nn.functional.interpolate(output.get('logits').detach(), size=image.shape[-2:], mode="bilinear", align_corners=False).argmax(dim=1).numpy()[0]
        return output

# Auxiliary method to generate streaming output
def gen_frames():
    while camera.isOpened():
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Get streaming image and store it in cache
            # Decode it as array
            frame = cv2.resize(frame, (512, 512), interpolation = cv2.INTER_AREA)
            frame = ToTensor()(frame)
            output = inference(model, feature_extractor, frame)
            my_cm = matplotlib.cm.get_cmap('jet')
            mapped_data = my_cm(output/18, bytes=True)
            frame_output = cv2.imencode('.png', mapped_data)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_output + b'\r\n')  # concat frame one by one and show result

#Initialize the Flask app
app = Flask(__name__)

# Acessing camera
## for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 
## for local webcam use cv2.VideoCapture(0)
camera = cv2.VideoCapture(0)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Control de código
if __name__ == '__main__':
    # Prepare stuff
    idx2tag, feature_extractor, model = load_checkpoint()
    # El método "run" activa el servicio, y espera a ser llamado
    ## El host "0.0.0.0" indica que el servicio va a ser desplegado
    ## en máquina local, y que puede recibir peticiones mediante el
    ## puerto 8080. Finalmente, activar el debug nos permite ver
    ## cambios conforme los guardamos.
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))