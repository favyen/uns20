import json
import sys

in_fname = sys.argv[1]
out_fname = sys.argv[2]

with open(in_fname, 'r') as f:
	detections = json.load(f)

# get tracks
track_map = {}
for dlist in detections:
	if dlist is None:
		continue
	for detection in dlist:
		track_id = detection['track_id']
		if track_id not in track_map:
			track_map[track_id] = []
		track_map[track_id].append(detection)

# interpolate tracks
ndetections = [[] for _ in detections]
for track in track_map.values():
	ntrack = []
	for detection in track:
		if len(ntrack) > 0:
			prev = ntrack[-1]
			next = detection
			jump = next['frame_idx'] - prev['frame_idx']
			for i in range(1, jump):
				prev_weight = float(jump-i) / float(jump)
				next_weight = float(i) / float(jump)
				interp = {
					'track_id': prev['track_id'],
					'frame_idx': prev['frame_idx']+i,
				}
				for k in ['left', 'top', 'right', 'bottom']:
					interp[k] = int(prev[k]*prev_weight + next[k]*next_weight)
				ntrack.append(interp)
		ntrack.append(detection)

	for detection in ntrack:
		ndetections[detection['frame_idx']].append(detection)

with open(out_fname, 'w') as f:
	json.dump(ndetections, f)
