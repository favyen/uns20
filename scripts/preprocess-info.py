import json
import math
import multiprocessing
import numpy
import os, os.path
import pickle
import skimage.io
import skimage.transform
import sys

sys.path.append('.')
import geom

dataset = sys.argv[1]
data_path = sys.argv[2]
nthreads = int(sys.argv[3])

ORIG_WIDTH = 1920
ORIG_HEIGHT = 1080
SKIP = 1
FRAME_SCALE = 1
CROP_SIZE = 64

if dataset == 'pathtrack':
	LABELS = [label for label in os.listdir(data_path + '/pathtrack/frames/')]
	FRAME_PATH = data_path + '/pathtrack/frames/{}/'
	DETECTION_PATH = data_path + '/pathtrack/json/{}.json'
	PICKLE_PATH = data_path + '/pathtrack/pickle-info/{}.pkl'
elif dataset == 'yt-walking':
	LABELS = [label for label in os.listdir(data_path + '/yt-walking/frames/')]
	FRAME_PATH = data_path + '/yt-walking/frames/{}/'
	DETECTION_PATH = data_path + '/yt-walking/json/{}.json'
	PICKLE_PATH = data_path + '/yt-walking/pickle-info/{}.pkl'
elif dataset == 'mot17':
	LABELS = [label for label in os.listdir(data_path + '/mot17/frames/')]
	FRAME_PATH = data_path + '/mot17/frames/{}/'
	DETECTION_PATH = data_path + '/mot17/json/{}.json'
	PICKLE_PATH = data_path + '/mot17/pickle-info/{}.pkl'

os.makedirs(os.path.dirname(PICKLE_PATH), exist_ok=True)

def get_frame_fname(frame_idx):
	s = str(frame_idx)
	while len(s) < 6:
		s = '0' + s
	return s + '.jpg'

def to_rect(detection):
	return geom.Rectangle(
		geom.Point(detection['left'], detection['top']),
		geom.Point(detection['right'], detection['bottom']),
	)

MAX_MATCH_AGE = 5
def get_potential_matches(detections, first_frame, last_frame):
	# from detection_idx in frame #first_idx to iterable of matching tuples (frame, det_idx)
	cur_matches = {}
	for idx in range(len(detections[first_frame])):
		cur_matches[idx] = [(first_frame, idx)]

	for right_frame in range(first_frame+1, last_frame+1):
		# list the detections we need to match
		check_set = set()
		for l in cur_matches.values():
			check_set.update(l)

		connections = {}
		for left_frame, left_idx in check_set:
			connections[(left_frame, left_idx)] = []

			for right_idx in range(len(detections[right_frame])):
				rect1 = to_rect(detections[left_frame][left_idx])
				rect2 = to_rect(detections[right_frame][right_idx])
				intersect_area = rect1.intersection(rect2).area()
				if intersect_area < 0:
					intersect_area = 0
				union_area = rect1.area() + rect2.area() - intersect_area
				iou_score = float(intersect_area) / float(union_area)
				if iou_score > 0.1:
					connections[(left_frame, left_idx)].append((right_frame, right_idx))

		for idx in cur_matches:
			new_matches = set()
			for left_frame, left_idx in cur_matches[idx]:
				new_matches.update(connections[(left_frame, left_idx)])
				if right_frame - left_frame < MAX_MATCH_AGE:
					new_matches.add((left_frame, left_idx))
			cur_matches[idx] = new_matches

	final_matches = {}
	for idx, matches in cur_matches.items():
		final_matches[idx] = [right_idx for right_frame, right_idx in matches if right_frame == last_frame]
	return final_matches

def zip_frame_info(detections, label, frame_idx):
	if not detections:
		return []
	frame_path = FRAME_PATH.format(label)
	im = skimage.io.imread('{}/{}'.format(frame_path, get_frame_fname(frame_idx)))
	im_bounds = geom.Rectangle(
		geom.Point(0, 0),
		geom.Point(im.shape[0], im.shape[1])
	)
	info = []
	for idx, detection in enumerate(detections):
		rect = geom.Rectangle(
			geom.Point(detection['top']/FRAME_SCALE, detection['left']/FRAME_SCALE),
			geom.Point(detection['bottom']/FRAME_SCALE, detection['right']/FRAME_SCALE)
		)
		if rect.lengths().x < 4 or rect.lengths().y < 4:
			continue
		crop = im[rect.start.x:rect.end.x, rect.start.y:rect.end.y, :]
		resize_factor = min([float(CROP_SIZE) / crop.shape[0], float(CROP_SIZE) / crop.shape[1]])
		resize_shape = [int(crop.shape[0] * resize_factor), int(crop.shape[1] * resize_factor)]
		if resize_shape[0] == 0 or resize_shape[1] == 0:
			continue
		crop = (skimage.transform.resize(crop, resize_shape)*255).astype('uint8')
		fix_crop = numpy.zeros((CROP_SIZE, CROP_SIZE, 3), dtype='uint8')
		fix_crop[0:crop.shape[0], 0:crop.shape[1], :] = crop
		detection['width'] = float(detection['right']-detection['left'])/ORIG_WIDTH
		detection['height'] = float(detection['bottom']-detection['top'])/ORIG_HEIGHT
		info.append((detection, fix_crop, idx))
	return info

def process(label):
	pickle_path = PICKLE_PATH.format(label)
	if os.path.exists(pickle_path):
		return
	print('reading from {}'.format(label))
	with open(DETECTION_PATH.format(label), 'r') as f:
		detections = json.load(f)

	if not detections:
		return

	frame_infos = {}
	#matches = {}
	for frame_idx in range(0, len(detections), SKIP):
		if frame_idx % 30000 > 10000:
			continue
		print(label, frame_idx)
		if not detections[frame_idx]:
			continue
		frame_infos[frame_idx] = zip_frame_info(detections[frame_idx], label, frame_idx)
		#for match_len in [10, 15, 25, 35, 45, 55, 65]:
		#	frame_range = range(frame_idx, frame_idx+match_len+1)
		#	if not all([detections[i] is not None and len(detections[i]) > 0 for i in frame_range]):
		#		continue
		#	for i in frame_range:
		#		frame_infos[i] = zip_frame_info(detections[i], label, i)
		#	matches[(frame_idx, frame_idx+match_len)] = get_potential_matches(detections, frame_idx, frame_idx+match_len)

	with open(pickle_path, 'wb') as f:
		pickle.dump(frame_infos, f)

p = multiprocessing.Pool(nthreads)
p.map(process, LABELS)
p.close()
