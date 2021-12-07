import geom
import model

import json
import math
import numpy
import os
import pickle
import random
import skimage.io, skimage.transform
import sys
import tensorflow as tf
import time

data_path = sys.argv[1]
model_path = sys.argv[2]

ORIG_WIDTH = 1920
ORIG_HEIGHT = 1080
FRAME_SCALE = 1
CROP_SIZE = 64
MATCH_LENGTHS = [4, 16]
ADD_NEGATIVES = True
MODE = 'imsp-longim'

SKIPS = [1]
DATASETS = [
	(
		os.listdir(data_path + '/yt-walking/frames/'),
		data_path + '/yt-walking/frames/{}/',
		data_path + '/yt-walking/pickle-info/{}.pkl',
		data_path + '/yt-walking/pickle-info/{}.matches.json',
		2.0,
		[1, 2],
	),
	(
		os.listdir(data_path + '/pathtrack/frames/'),
		data_path + '/pathtrack/frames/{}/',
		data_path + '/pathtrack/pickle-info/{}.pkl',
		data_path + '/pathtrack/pickle-info/{}.matches.json',
		1.5,
		[2, 4],
	),
	(
		os.listdir(data_path + '/mot17/frames/'),
		data_path + '/mot17/frames/{}/',
		data_path + '/mot17/pickle-info/{}.pkl',
		data_path + '/mot17/pickle-info/{}.matches.json',
		1.0,
		[2, 4],
	),
]
val_fn = lambda example: hash(example[4]) % 20 == 0 and 'MOT' not in example[4]

def get_frame_fname(frame_idx):
	s = str(frame_idx)
	while len(s) < 6:
		s = '0' + s
	return s + '.jpg'

def get_loc(detection):
	cx = (detection['left'] + detection['right']) / 2
	cy = (detection['top'] + detection['bottom']) / 2
	cx = float(cx) / ORIG_WIDTH
	cy = float(cy) / ORIG_HEIGHT
	return cx, cy

def to_rect(detection):
	return geom.Rectangle(
		geom.Point(detection['left'], detection['top']),
		geom.Point(detection['right'], detection['bottom']),
	)

def get_stuff(infos, matches):
	def per_info(info):
		images = []
		boxes = numpy.zeros((len(info), 4), dtype='float32')
		for i, (detection, crop, _) in enumerate(info):
			images.append(crop)
			cx, cy = get_loc(detection)
			boxes[i, :] = [cx, cy, detection['width'], detection['height']]
		detections = [get_loc(detection) for detection, _, _ in info]
		return images, boxes, detections, len(info)

	all_images = []
	all_boxes = []
	all_detections = []
	all_counts = []
	for i, info in enumerate(infos):
		images, boxes, detections, count = per_info(info)
		all_images.append(images)
		all_boxes.append(boxes)
		all_detections.append(detections)
		all_counts.append(count)

	all_masks = []
	for i, match_len in enumerate(MATCH_LENGTHS):
		last_idx = match_len
		mask = numpy.zeros((len(infos[0]), len(infos[last_idx])+1), dtype='float32')
		mask[:, len(infos[last_idx])] = 1
		first_map = {}
		for j, (_, _, orig_idx) in enumerate(infos[0]):
			first_map[orig_idx] = j
		last_map = {}
		for j, (_, _, orig_idx) in enumerate(infos[last_idx]):
			last_map[orig_idx] = j
		for left_idx in matches[i]:
			if left_idx not in first_map:
				continue
			for right_idx in matches[i][left_idx]:
				if right_idx not in last_map:
					continue
				mask[first_map[left_idx], last_map[right_idx]] = 1
		all_masks.append(mask.flatten())

	return all_images, all_boxes, all_detections, all_counts, all_masks

print('loading infos and matches')
all_frame_data = {}
for labels, frame_tmpl, pickle_tmpl, match_tmpl, detection_scale, _ in DATASETS:
	for label in labels:
		pickle_path = pickle_tmpl.format(label)
		match_path = match_tmpl.format(label)
		print('... {} (pickle)'.format(label))
		with open(pickle_path, 'rb') as f:
			frame_infos = pickle.load(f, encoding='latin1')
		for info_list in frame_infos.values():
			for info in info_list:
				info[0]['left'] *= detection_scale
				info[0]['top'] *= detection_scale
				info[0]['right'] *= detection_scale
				info[0]['bottom'] *= detection_scale
				info[0]['width'] *= detection_scale
				info[0]['height'] *= detection_scale
		print('... {} (matches)'.format(label))
		with open(match_path, 'r') as f:
			raw_matches = json.load(f, encoding='latin1')
			frame_matches = {}
			for match_len in raw_matches:
				frame_matches[int(match_len)] = {}
				for frame_idx in raw_matches[match_len]:
					frame_matches[int(match_len)][int(frame_idx)] = {}
					for left_idx in raw_matches[match_len][frame_idx]:
						frame_matches[int(match_len)][int(frame_idx)][int(left_idx)] = raw_matches[match_len][frame_idx][left_idx]
		all_frame_data[label] = (frame_infos, frame_matches)

print('preparing random info generator')
labels_and_weights = [(label, len(all_frame_data[label][0])) for label in all_frame_data.keys()]
def get_random_info(exclude_label):
	labels = [label for label in all_frame_data.keys() if label != exclude_label]
	weights = [len(all_frame_data[label][0]) for label in labels]
	weight_sum = sum(weights)
	weights = [float(x)/float(weight_sum) for x in weights]
	while True:
		label = numpy.random.choice(labels, p=weights)
		frame_infos = all_frame_data[label][0]
		frame_idx = random.choice(list(frame_infos.keys()))
		if len(frame_infos[frame_idx]) > 4:
			return frame_infos[frame_idx]

# each example is tuple (images, boxes, n_image, label, frame_idx, skip)
print('extracting examples')
all_examples = []
for labels, frame_tmpl, _, _, _, skips in DATASETS:
	for label in labels:
		frame_path = frame_tmpl.format(label)
		frame_infos, frame_matches = all_frame_data[label]

		for i, frame_idx in enumerate(frame_infos.keys()):
			print('...', label, i, len(frame_infos))

			skip = random.choice(skips)
			match_lengths = [skip*match_len for match_len in MATCH_LENGTHS]

			infos = [frame_infos.get(frame_idx+l*skip, None) for l in range(model.SEQ_LEN)]
			if any([(info is None or len(info) == 0) for info in infos]):
				continue
			elif any([frame_idx not in frame_matches[match_len] for match_len in match_lengths]):
				continue

			if ADD_NEGATIVES:
				neg_info = get_random_info(label)
			else:
				neg_info = []

			matches = [frame_matches[match_len][frame_idx] for match_len in match_lengths]
			images, boxes, detections, counts, mask = get_stuff(infos + [neg_info], matches)
			all_examples.append((
				images, boxes, counts, mask,
				label, frame_idx, detections, frame_path, skip,
			))

random.shuffle(all_examples)
val_examples = [example for example in all_examples if val_fn(example)]
if len(val_examples) > 1024:
	val_examples = random.sample(val_examples, 1024)
train_examples = [example for example in all_examples if not val_fn(example) and min(example[2][:-1]) >= 6]

best_loss = None

def train(learning_rate, num_epochs):
	global best_loss

	print('training mode={} at lr={} for {} epochs'.format(MODE, learning_rate, num_epochs))
	for epoch in range(num_epochs):
		start_time = time.time()
		train_losses = []
		for _ in range(2048//model.BATCH_SIZE):
			if MODE == 'imsp-finesp':
				match_len = max(MATCH_LENGTHS)
			else:
				match_len = random.choice(MATCH_LENGTHS)

			batch = []
			for example in random.sample(train_examples, model.BATCH_SIZE):
				imlists = example[0][0:match_len+1] + [example[0][model.SEQ_LEN]]
				boxlists = example[1][0:match_len+1] + [example[1][model.SEQ_LEN]]
				counts = example[2][0:match_len+1] + [example[2][model.SEQ_LEN]]
				mask = example[3][MATCH_LENGTHS.index(match_len)]
				batch.append((imlists, boxlists, counts, mask))

			imlists = [imlist for example in batch for imlist in example[0]]
			boxlists = [boxlist for example in batch for boxlist in example[1]]
			counts = [[] for _ in range(len(batch))]
			for i, example in enumerate(batch):
				counts[i] = example[2][0:match_len+1]
				while len(counts[i]) < model.SEQ_LEN:
					counts[i].append(0)
				counts[i].append(example[2][-1])

			images = [im for imlist in imlists for im in imlist]
			boxes = [box for boxlist in boxlists for box in boxlist]

			masks = numpy.concatenate([example[3] for example in batch], axis=0)
			feed_dict = {
				m.raw_images: images,
				m.input_boxes: boxes,
				m.n_image: counts,
				m.input_masks: masks,
				m.match_length: match_len,
				m.is_training: True,
				m.learning_rate: learning_rate,
			}
			if MODE == 'imsp-longim':
				_, loss = session.run([m.longim_optimizer, m.longim_loss], feed_dict=feed_dict)
			elif MODE == 'imsp-finesp':
				_, loss = session.run([m.finesp_optimizer, m.finesp_loss], feed_dict=feed_dict)
			train_losses.append(loss)
		train_loss = numpy.mean(train_losses)
		train_time = time.time()

		val_losses = []
		for i in range(0, len(val_examples), model.BATCH_SIZE):
			batch = val_examples[i:i+model.BATCH_SIZE]
			images = [im for example in batch for imlist in example[0] for im in imlist]
			boxes = [box for example in batch for boxlist in example[1] for box in boxlist]
			counts = [example[2] for example in batch]
			masks = numpy.concatenate([example[3][-1] for example in batch], axis=0)
			feed_dict = {
				m.raw_images: images,
				m.input_boxes: boxes,
				m.n_image: counts,
				m.input_masks: masks,
				m.match_length: model.SEQ_LEN-1,
				m.is_training: False,
			}
			if MODE == 'imsp-longim':
				loss = session.run(m.longim_loss, feed_dict=feed_dict)
			elif MODE == 'imsp-finesp':
				loss = session.run(m.finesp_loss, feed_dict=feed_dict)
			val_losses.append(loss)

		val_loss = numpy.mean(val_losses)
		val_time = time.time()

		print('iteration {}: train_time={}, val_time={}, train_loss={}, val_loss={}/{}'.format(epoch, int(train_time - start_time), int(val_time - train_time), train_loss, val_loss, best_loss))

		if best_loss is None or val_loss < best_loss:
			best_loss = val_loss
			m.saver.save(session, model_path)


print('initializing model: longim')
m = model.Model(options={'mode': MODE})
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
session.run(m.init_op)

train(1e-3, 200)
train(1e-4, 200)
train(1e-5, 200)

print('initializing model: finesp')
MODE = 'imsp-finesp'
session.close()
m = model.Model(options={'mode': MODE})
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
m.saver.restore(session, model_path)

train(1e-3, 200)
train(1e-4, 200)
train(1e-5, 200)

print('done')
