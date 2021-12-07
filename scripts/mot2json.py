import json
import subprocess
import sys

data_path = sys.argv[1]
split = sys.argv[2]

if split == 'train':
	LABELS = ['02', '04', '05', '09', '10', '11', '13']
elif split == 'test':
	LABELS = ['01', '03', '06', '07', '08', '12', '14']

for label in LABELS:
	detections = []
	with open(data_path + '/mot17/{}/MOT17-{}-SDP/det/det.txt'.format(split, label), 'r') as f:
		lines = f.readlines()
	for line in lines:
		parts = line.strip().split(',')
		if len(parts) < 7:
			continue
		frame_idx = int(parts[0])
		track_id = int(parts[1])
		left = int(float(parts[2]))
		top = int(float(parts[3]))
		right = left + int(float(parts[4]))
		bottom = top + int(float(parts[5]))
		score = float(parts[6])
		if score < 0.6:
			continue
		while frame_idx >= len(detections):
			detections.append([])
		detections[frame_idx].append({
			'frame_idx': frame_idx,
			'track_id': track_id,
			'left': left,
			'top': top,
			'right': right,
			'bottom': bottom,
		})

	fname = data_path + '/mot17/{}/MOT17-{}-SDP/det/det-filter60.json'.format(split, label)
	with open(fname, 'w') as f:
		json.dump(detections, f)
	subprocess.call([
		'python', 'scripts/filter_small.py',
		fname,
		data_path + '/mot17/{}/MOT17-{}-SDP/img1/'.format(split, label),
		fname,
	])
