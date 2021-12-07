import json
import os
import subprocess
import sys

data_path = sys.argv[1]

os.makedirs(data_path + 'pathtrack/json/', exist_ok=True)

labels = os.listdir(data_path + '/pathtrack/train/')
for i, label in enumerate(labels):
	print(label, i, len(labels))

	detections = []
	with open(data_path + '/pathtrack/train/{}/det/det_rcnn.txt'.format(label), 'r') as f:
		lines = [line.strip() for line in f.readlines() if line.strip()]
	for line in lines:
		parts = line.split(',')
		frame_idx = int(float(parts[0]))
		while len(detections) <= frame_idx:
			detections.append([])
		left = int(float(parts[2]))
		top = int(float(parts[3]))
		right = left+int(float(parts[4]))
		bottom = top+int(float(parts[5]))
		score = float(parts[6])
		if score < 0.5:
			continue
		detections[frame_idx].append({
			'left': left,
			'top': top,
			'right': right,
			'bottom': bottom,
			'frame_idx': frame_idx,
			'track_id': -1,
		})

	with open(data_path + '/pathtrack/json/{}.json'.format(label), 'w') as f:
		json.dump(detections, f)
