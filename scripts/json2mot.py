import json
import os, os.path
import sys
import subprocess

data_path = sys.argv[1]
split = sys.argv[2]
out_path = sys.argv[3]

MODE = 'uns20'

labels = [fname for fname in os.listdir(data_path + '/mot17/{}/'.format(split))]
labels = [label for label in labels if 'SDP' in label]
labels.sort()

for label in labels:
	subprocess.call([
		'python', 'scripts/filter_short.py',
		data_path + '/mot17/{}/{}/det/{}.json'.format(split, label, MODE),
		data_path + '/mot17/{}/{}/det/{}-noshort.json'.format(split, label, MODE),
	])

for label in labels:
	subprocess.call([
		'python', 'scripts/interpolate.py',
		data_path + '/mot17/{}/{}/det/{}-noshort.json'.format(split, label, MODE),
		data_path + '/mot17/{}/{}/det/{}-interp.json'.format(split, label, MODE),
	])

for label in labels:
	with open(data_path + '/mot17/{}/{}/det/{}-interp.json'.format(split, label, MODE), 'r') as f:
		detections = json.load(f)

	lines = []
	for frame_idx, dlist in enumerate(detections):
		if dlist is None:
			continue
		for d in dlist:
			w = d['right'] - d['left']
			h = d['bottom'] - d['top']
			line = "{},{},{},{},{},{},-1,-1,-1,-1".format(d['frame_idx'], d['track_id']+1, d['left'], d['top'], w, h)
			lines.append(line)
	lines.append("")
	with open(os.path.join(out_path, '{}.txt'.format(label)), 'w') as f:
		f.write("\n".join(lines))
