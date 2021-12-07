import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class InferenceConfig(coco.CocoConfig):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = [
	'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
	'bus', 'train', 'truck', 'boat', 'traffic light',
	'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
	'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
	'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
	'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
	'kite', 'baseball bat', 'baseball glove', 'skateboard',
	'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
	'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
	'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
	'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
	'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
	'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
	'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
	'teddy bear', 'hair drier', 'toothbrush',
]

data_path = sys.argv[1]

import json
import os
import subprocess
FRAME_PATH = data_path + '/yt-walking/frames/'
JSON_PATH = data_path + '/yt-walking/json/'
BATCH_SIZE = 1
labels = os.listdir(FRAME_PATH)
labels.sort()
for label in labels[0:1]:
	print('processing', label)
	im_path = FRAME_PATH + label + '/'
	fnames = os.listdir(im_path)
	detections = []
	for i in range(0, len(fnames), BATCH_SIZE):
		print(label, i, len(fnames))
		batch = fnames[i:i+BATCH_SIZE]
		ims = [skimage.io.imread(im_path + fname) for fname in batch]
		results = model.detect(ims)
		for j in range(len(batch)):
			frame_idx = int(batch[j].split('.')[0])
			while len(detections) <= frame_idx:
				detections.append([])
			for roi, class_id in zip(results[j]['rois'], results[j]['class_ids']):
				if int(class_id) != 1:
					continue
				detections[frame_idx].append({
					'frame_idx': frame_idx,
					'left': int(roi[1]),
					'top': int(roi[0]),
					'right': int(roi[3]),
					'bottom': int(roi[2]),
					'track_id': -1,
				})

	json_fname = JSON_PATH + label + '.json'
	with open(json_fname, 'w') as f:
		json.dump(detections, f)
