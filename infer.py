import geom
import model

import json
import numpy
import math
import os
import skimage.io, skimage.transform
import sys
import tensorflow as tf
import time

MODEL_PATH = sys.argv[1]
data_path = sys.argv[2]

model.BATCH_SIZE = 1
model.SEQ_LEN = 2

SKIP = 2
MAX_AGE = 10
MODE = 'imsp'

LABELS = ['MOT17-{}-SDP'.format(x) for x in ['01', '03', '06', '07', '08', '12', '14']]
DETECTION_PATH = data_path + '/mot17/test/{}/det/det-filter60.json'
FRAME_PATH = data_path + '/mot17/test/{}/img1/'
OUT_PATH = data_path + '/mot17/test/{}/det/uns20.json'

ORIG_WIDTH = 1920
ORIG_HEIGHT = 1080
DETECTION_SCALE = 1
FRAME_SCALE = 1
CROP_SIZE = 64
HIDDEN_SIZE = 4*64

print('initializing model')
m = model.Model(options={'mode': MODE})
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

m.saver.restore(session, MODEL_PATH)

def get_frame_fname(frame_idx):
	s = str(frame_idx)
	while len(s) < 6:
		s = '0' + s
	return s + '.jpg'

for label in LABELS:
	detection_path = DETECTION_PATH.format(label)
	print('loading detections from {}'.format(detection_path))
	with open(detection_path, 'r') as f:
		raw_detections = json.load(f)

	# auto-detect im width/height
	for frame_idx, dlist in enumerate(raw_detections):
		if not dlist or len(dlist) == 0:
			continue
		im = skimage.io.imread('{}/{}'.format(FRAME_PATH.format(label), get_frame_fname(frame_idx)))
		im_bounds = geom.Rectangle(geom.Point(0, 0), geom.Point(im.shape[1]*FRAME_SCALE, im.shape[0]*FRAME_SCALE))
		break

	detections = [None for _ in range(len(raw_detections))]
	for frame_idx, dlist in enumerate(raw_detections):
		if not dlist or frame_idx % SKIP != 0:
			continue
		detections[frame_idx] = []
		for i, d in enumerate(dlist):
			rect = geom.Rectangle(
				geom.Point(d['left']//DETECTION_SCALE, d['top']//DETECTION_SCALE),
				geom.Point(d['right']//DETECTION_SCALE, d['bottom']//DETECTION_SCALE)
			)
			rect = im_bounds.clip_rect(rect)
			if rect.lengths().x < 4 or rect.lengths().y < 4:
				continue
			nd = {
				'left': rect.start.x,
				'top': rect.start.y,
				'right': rect.end.x,
				'bottom': rect.end.y,
				'frame_idx': d['frame_idx'],
			}
			detections[frame_idx].append(nd)

	def zip_frame_info(detections, frame_idx):
		im = skimage.io.imread('{}/{}'.format(FRAME_PATH.format(label), get_frame_fname(frame_idx)))
		im_bounds = geom.Rectangle(
			geom.Point(0, 0),
			geom.Point(im.shape[0], im.shape[1])
		)
		info = []
		for detection in detections:
			rect = geom.Rectangle(
				geom.Point(detection['top']//FRAME_SCALE, detection['left']//FRAME_SCALE),
				geom.Point(detection['bottom']//FRAME_SCALE, detection['right']//FRAME_SCALE)
			)
			crop = im[rect.start.x:rect.end.x, rect.start.y:rect.end.y, :]
			resize_factor = min([float(CROP_SIZE) / crop.shape[0], float(CROP_SIZE) / crop.shape[1]])
			crop = (skimage.transform.resize(crop, [int(crop.shape[0] * resize_factor), int(crop.shape[1] * resize_factor)])*255).astype('uint8')
			fix_crop = numpy.zeros((CROP_SIZE, CROP_SIZE, 3), dtype='uint8')
			fix_crop[0:crop.shape[0], 0:crop.shape[1], :] = crop
			detection['width'] = float(detection['right']-detection['left'])/ORIG_WIDTH
			detection['height'] = float(detection['bottom']-detection['top'])/ORIG_HEIGHT
			info.append((detection, fix_crop))
		return info

	def get_loc(detection):
		cx = (detection['left'] + detection['right']) / 2
		cy = (detection['top'] + detection['bottom']) / 2
		cx = float(cx) / ORIG_WIDTH
		cy = float(cy) / ORIG_HEIGHT
		return cx, cy

	def get_stuff(infos):
		def per_info(info):
			images = []
			boxes = []
			for i, (detection, crop) in enumerate(info):
				images.append(crop)
				cx, cy = get_loc(detection)
				boxes.append([cx, cy, detection['width'], detection['height']])
			detections = [get_loc(detection) for detection, _ in info]
			return images, boxes, detections, len(info)

		all_images = []
		all_boxes = []
		all_detections = []
		all_counts = []
		for info in infos:
			images, boxes, detections, count = per_info(info)
			all_images.extend(images)
			all_boxes.extend(boxes)
			all_detections.append(detections)
			all_counts.append(count)

		return all_images, all_boxes, all_detections, all_counts

	def softmax(X, theta = 1.0, axis = None):
		y = numpy.atleast_2d(X)
		if axis is None:
		    axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
		y = y * float(theta)
		y = y - numpy.expand_dims(numpy.max(y, axis = axis), axis)
		y = numpy.exp(y)
		ax_sum = numpy.expand_dims(numpy.sum(y, axis = axis), axis)
		p = y / ax_sum
		if len(X.shape) == 1: p = p.flatten()
		return p

	# list of objects (id, detection_idx in latest frame, prev_hidden, time since last match)
	# note: detection_idx should be len(info)+1 for the terminal vertex
	active_objects = None
	track_counter = 0
	for frame_idx in range(0, len(detections)-SKIP, SKIP):
		if not detections[frame_idx] or not detections[frame_idx+SKIP]:
			active_objects = None
			continue

		print(frame_idx, len(detections))
		info1 = zip_frame_info(detections[frame_idx], frame_idx)
		info2 = zip_frame_info(detections[frame_idx+SKIP], frame_idx+SKIP)

		if len(info1) == 0 or len(info2) == 0:
			active_objects = None
			continue

		images1, boxes1, _, counts1 = get_stuff([info1])
		images2, boxes2, _, counts2 = get_stuff([info2])

		if active_objects is None:
			active_objects = []
			for left_idx in range(len(info1)):
				active_objects.append((
					track_counter,
					left_idx,
					numpy.zeros((HIDDEN_SIZE,), dtype='float32'),
					0,
					[images1[left_idx]],
				))
				detections[frame_idx][left_idx]['track_id'] = track_counter
				track_counter += 1

		'''
		outputs_raw, out_mat, cur_hidden, out_logits, mat_finesp, mat_longim = session.run([m.out_mat_reweight, m.out_mat, m.out_hidden, m.out_logits_finesp, m.out_mat_finesp, m.out_mat_longim], feed_dict=feed_dict)

		# take maximum in outputs_raw along the active indices
		outputs = numpy.zeros((len(active_objects), len(info2)+1), dtype='float32')
		for i, obj in enumerate(active_objects):
			cur_finesp = mat_finesp[active_indices[i], :].max(axis=0)
			cur_longim = mat_longim[active_indices[i], :].max(axis=0)
			outputs[i, :] = cur_finesp + cur_longim
			#outputs[i, 0:len(info2)] = outputs_raw[active_indices[i], 0:len(info2)].max(axis=0)
			#outputs[i, len(info2)] = outputs_raw[active_indices[i], len(info2)].min()
		'''

		if MODE == 'imsp' or MODE == 'finesp' or MODE == 'longim':
			# flatten the active objects since each object may have multiple images
			flat_images = []
			flat_boxes = []
			flat_hidden = []
			active_indices = {}
			for i, obj in enumerate(active_objects):
				active_indices[i] = []
				for j in [1, 2, 4, 8, 16]:
				#for j in range(1, len(obj[4])+1, len(obj[4])//5+1):
					if len(obj[4]) < j:
						continue
					# use image from stored history, but use current box
					active_indices[i].append(len(flat_images))
					flat_images.append(obj[4][-j])
					if obj[1] < len(info1):
						flat_boxes.append(boxes1[obj[1]])
					else:
						flat_boxes.append(numpy.zeros((4,), dtype='float32'))
					flat_hidden.append(obj[2])

			feed_dict = {
				m.raw_images: flat_images + images2,
				m.input_boxes: flat_boxes + boxes2,
				m.n_image: [[len(flat_images), len(images2), 0]],
				m.is_training: False,
				m.infer_sel: range(len(flat_images)),
				m.infer_hidden: flat_hidden,
			}

			longim_logits, finesp_logits, pre_cur_hidden = session.run([m.out_logits_longim, m.out_logits_finesp, m.out_hidden], feed_dict=feed_dict)
			longim_out_logits = numpy.zeros((len(active_objects), len(info2)+1), dtype='float32')
			finesp_out_logits = numpy.zeros((len(active_objects), len(info2)+1), dtype='float32')
			cur_hidden = numpy.zeros((len(active_objects), len(info2)+1, HIDDEN_SIZE), dtype='float32')
			for i, obj in enumerate(active_objects):
				longim_out_logits[i, 0:len(info2)] = longim_logits[active_indices[i], 0:len(info2)].mean(axis=0)
				longim_out_logits[i, len(info2)] = longim_logits[active_indices[i], len(info2)].min()
				finesp_out_logits[i, 0:len(info2)] = finesp_logits[active_indices[i], 0:len(info2)].mean(axis=0)
				finesp_out_logits[i, len(info2)] = finesp_logits[active_indices[i], len(info2)].min()
				cur_hidden[i, :, :] = pre_cur_hidden[active_indices[i][0], :, :]
			#longim_mat = softmax(longim_out_logits, axis=1)
			#finesp_mat = softmax(finesp_out_logits, axis=1)
			longim_mat = numpy.minimum(softmax(longim_out_logits, axis=0), softmax(longim_out_logits, axis=1))
			finesp_mat = numpy.minimum(softmax(finesp_out_logits, axis=0), softmax(finesp_out_logits, axis=1))
			outputs = numpy.minimum(longim_mat, finesp_mat)
			#outputs = numpy.minimum(longim_out_logits, finesp_out_logits)
			#outputs = (longim_out_logits+finesp_out_logits)/2
			if MODE == 'finesp':
				outputs = finesp_mat
			elif MODE == 'longim':
				outputs = longim_mat
		else:
			feed_dict = {
				m.raw_images: images1 + images2,
				m.input_boxes: boxes1 + boxes2,
				m.is_training: False,
				m.infer_sel: [obj[1] for obj in active_objects],
				m.infer_hidden: [obj[2] for obj in active_objects],
			}
			if MODE == 'occl':
				feed_dict[m.a_counts] = [len(images1), len(images2)]
			else:
				feed_dict[m.n_image] = [[len(images1), len(images2), 0]]
			outputs, out_mat, out_logits, cur_hidden = session.run([m.out_mat_reweight, m.out_mat, m.out_logits, m.out_hidden], feed_dict=feed_dict)
			outputs = out_mat

		# vote on best next frame: idx1->(output,idx2)
		votes = {}
		for i in range(len(active_objects)):
			for j in range(len(info2)+1):
				output = outputs[i, j]
				#if j == len(info2) and out_logits[active_indices[i][0], :].argmax() == len(info2):
				#if j == len(info2) and longim_out_logits[:, outputs[i, :].argmax()].argmax() != i:
				#if j == len(info2) and longim_out_logits[i, :].max() < 1:
				if MODE == 'imsp' and j != len(info2) and (longim_out_logits[i, j] < 0 or finesp_out_logits[i, j] < 0):
					output = -100.0
				elif MODE == 'finesp' and j != len(info2) and finesp_out_logits[i, j] < 0:
					output = -100.0
				#if j == len(info2):
				#	output = -2
				if i not in votes or output > votes[i][0]:
					if j < len(info2):
						votes[i] = (output, j)
					else:
						votes[i] = (output, None)
		# group by receiver and vote on max idx2->idx1 to eliminate duplicates
		votes2 = {}
		for idx1, t in votes.items():
			output, idx2 = t
			if idx2 is not None and (idx2 not in votes2 or output > votes2[idx2][0]):
				votes2[idx2] = (output, idx1)
		forward_matches = {idx1: idx2 for (idx2, (_, idx1)) in votes2.items()}

		def get_hidden(idx1, idx2):
			if model.__name__ == 'occl3b_model':
				return cur_hidden[idx1, :]
			else:
				return cur_hidden[idx1, idx2, :]

		new_objects = []
		used_idx2s = set()
		for idx1, obj in enumerate(active_objects):
			if idx1 in forward_matches:
				idx2 = forward_matches[idx1]
				new_objects.append((
					obj[0],
					idx2,
					get_hidden(idx1, idx2),
					#numpy.zeros((64,), dtype='float32'),
					0,
					obj[4] + [images2[idx2]],
				))
				used_idx2s.add(idx2)
				detections[frame_idx+SKIP][idx2]['track_id'] = obj[0]
			elif obj[3] < MAX_AGE:
				idx2 = votes[idx1][1]
				if idx2 is None or True:
					idx2 = len(info2)
				new_objects.append((
					obj[0],
					idx2,
					get_hidden(idx1, idx2),
					#numpy.zeros((64,), dtype='float32'),
					obj[3]+1,
					obj[4],
				))

		for idx2 in range(len(info2)):
			if idx2 in used_idx2s:
				continue
			new_objects.append((
				track_counter,
				idx2,
				numpy.zeros((HIDDEN_SIZE,), dtype='float32'),
				0,
				[images2[idx2]],
			))
			detections[frame_idx+SKIP][idx2]['track_id'] = track_counter
			track_counter += 1

		active_objects = new_objects

	ndetections = [None for _ in detections]
	for frame_idx, dlist in enumerate(detections):
		if not dlist:
			continue
		dlist = [d for d in dlist if 'track_id' in d]
		if not dlist:
			continue
		ndetections[frame_idx] = dlist
	detections = ndetections

	with open(OUT_PATH.format(label), 'w') as f:
		json.dump(detections, f)
