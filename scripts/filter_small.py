import json
import os
import skimage.io
import sys

in_fname = sys.argv[1]
frame_path = sys.argv[2]
out_fname = sys.argv[3]

with open(in_fname, 'r') as f:
	detections = json.load(f)

frame_fname = [fname for fname in os.listdir(frame_path) if fname.endswith('.jpg')][0]
im = skimage.io.imread(frame_path + frame_fname)

ndetections = [[] for _ in detections]
for frame_idx, dlist in enumerate(detections):
	if dlist is None:
		continue
	for detection in dlist:
		if detection['left'] < 0:
			detection['left'] = 0
		if detection['right'] >= im.shape[1]:
			detection['right'] = im.shape[1]-1
		if detection['top'] < 0:
			detection['top'] = 0
		if detection['bottom'] >= im.shape[0]:
			detection['bottom'] = im.shape[0]-1

		if detection['right'] - detection['left'] <= 4:
			continue
		elif detection['bottom'] - detection['top'] <= 4:
			continue
		ndetections[frame_idx].append(detection)

with open(out_fname, 'w') as f:
	json.dump(ndetections, f)
