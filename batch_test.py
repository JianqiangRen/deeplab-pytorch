# coding=utf-8
# summary:
# author: xueluo
# date:
from __future__ import absolute_import, division, print_function
import glob
import os
import click
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
import random
from libs.models import *
from libs.utils import DenseCRF
import datetime

label2semantic = {}

config_path = 'configs/cocostuff164k.yaml'
model_path = 'data/models/coco/deeplabv1_resnet101/caffemodel/deeplabv2_resnet101_msc-cocostuff164k-100000.pth'
cuda = False
crf = False

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device

def get_classtable(CONFIG):
    with open(CONFIG.DATASET.LABELS) as f:
        classes = {}
        for label in f:
            label = label.rstrip().split("\t")
            classes[int(label[0])] = label[1].split(",")[0]
    return classes


def setup_postprocessor(CONFIG):
    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )
    return postprocessor


def preprocessing(image, device, CONFIG):
    # Resize
    scale = CONFIG.IMAGE.SIZE.TEST / max(image.shape[:2])
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    raw_image = image.astype(np.uint8)

    # Subtract mean values
    image = image.astype(np.float32)
    image -= np.array(
        [
            float(CONFIG.IMAGE.MEAN.B),
            float(CONFIG.IMAGE.MEAN.G),
            float(CONFIG.IMAGE.MEAN.R),
        ]
    )

    # Convert to torch.Tensor and add "batch" axis
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.to(device)

    return image, raw_image


def has_intersection(list_a, list_b):
    for a in list_a:
        if a in list_b:
            return True
    return False
    


def inference(model, image, raw_image=None, postprocessor=None):
    _, _, H, W = image.shape

    # Image -> Probability map
    logits = model(image)
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)[0]
    probs = probs.cpu().numpy()

    # Refine the prob map with CRF
    if postprocessor and raw_image is not None:
        probs = postprocessor(raw_image, probs)

    labelmap = np.argmax(probs, axis=0)

    return labelmap


if __name__ == "__main__":
    
    img_paths = glob.glob(os.path.join('landscape','*.png'))
    
    if len(img_paths) == 0:
        print("No images")
        exit()
    
    f = open(config_path)
    # Setup
    CONFIG = Dict(yaml.load(f))
    device = get_device(cuda)
    torch.set_grad_enabled(False)

    print(CONFIG.DATASET.LABELS)

    with open(CONFIG.DATASET.LABELS, 'r') as f:
        for line in f:
            labelsemantic = line.split()
            # print(labelsemantic)
            label2semantic[int(labelsemantic[0].strip())] = labelsemantic[1].strip()

    classes = get_classtable(CONFIG)
    postprocessor = setup_postprocessor(CONFIG) if crf else None

    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("Model:", CONFIG.MODEL.NAME)

    for img_path in img_paths:
        # Inference
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image, raw_image = preprocessing(image, device, CONFIG)
        labelmap = inference(model, image, raw_image, postprocessor)
        # print(type(labelmap), labelmap.dtype)
        # cv2.imwrite("labelmap.png", labelmap.astype(np.uint8))
        labels = np.unique(labelmap)
        print([label2semantic[label] for label in labels])

        fn = 'landscape_seg/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-") + str(
            random.randint(0, 10000000)) + ".png"


        sem_labels = [label2semantic[label] for label in labels]
        # text_str = ''
        # for i, t in  enumerate(sem_labels):
        #     text_str += ", " +t
        #     if i >0 and i % 4 == 0:
        #         text_str += '\n'
        #
        # y0, dy = 20, 25
        #
        # for i, txt in enumerate(text_str.split('\n')):
        #     y = y0 + i * dy
        #     cv2.putText(raw_image, txt, (0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 2)
        
        if has_intersection(sem_labels, ['wood', 'wall-wood', 'bench', 'car', 'bus']):
            continue
        
        if not has_intersection(sem_labels, ['sea', 'sea-other', 'sky', 'sky-other', 'water-ohter', 'cloud', 'tree', 'mountain',\
                                               'grass' ,'rock']):
            continue
        
        cv2.imwrite(fn,raw_image )
