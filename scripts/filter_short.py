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

ndetections = [[] for _ in detections]
for track in track_map.values():
	if len(track) <= 3:
		continue
	for detection in track:
		ndetections[detection['frame_idx']].append(detection)

with open(out_fname, 'w') as f:
	json.dump(ndetections, f)
