Self-Supervised Multi-Object Tracking with Cross-Input Consistency
------------------------------------------------------------------

UNS20 is the code for "Self-Supervised Multi-Object Tracking with Cross-Input Consistency" (NeurIPS 2021).
UNS20 is an approach for training a robust multi-object tracking model using
only an object detector and a large corpus of unlabeled video.


Installation
------------

Requires Tensorflow 1.15:

	pip install 'tensorflow<2.0' scikit-image

Download MOT17 dataset:

	mkdir /home/ubuntu/data/
	wget https://motchallenge.net/data/MOT17.zip
	unzip MOT17.zip
	mv MOT17 /home/ubuntu/data/mot17/

Download UNS20 model:

	wget https://favyen.com/files/uns20-model.zip
	mv model/ /home/ubuntu/model/


Inference
---------

For SDP detections:

	cd /path/to/uns20/
	python scripts/mot2json.py /home/ubuntu/data/ test
	python infer.py /home/ubuntu/model/model /home/ubuntu/data/

DPM and FRCNN detections have lower accuracy than SDP detections. Recent
methods universally perform regression and classification pre-processing steps.
Classification prunes incorrect input detections, while regression improves the
bounding box coordinates. These steps don't really make sense, since they use a
better detector to improve lower-quality detections. However, the steps are
needed to achieve performance comparable with other methods, since all methods
now use the same steps.

To apply UNS20 on DPM and FRCNN detections, it should be executed after the
regression and classification pre-processing steps from https://github.com/phil-bergmann/tracking_wo_bnw.

For the most informative comparison, we highly recommend comparing performance
only on the SDP detections, which have the highest accuracy. While evaluating
on lower-quality detections sounds like it could be useful, one would really be
evaluating the pre-processing steps more than the method itself.


Evaluation
----------

Convert from JSON to the TXT format:

	mkdir /home/ubuntu/outputs/
	python scripts/json2mot.py /home/ubuntu/data/ train /home/ubuntu/outputs/

Compare:

	pip install motmetrics
	python -m motmetrics.apps.eval_motchallenge /home/ubuntu/data/mot17/train/ /home/ubuntu/outputs/


Training
--------

First, obtain PathTrack and YT-Walking datasets:

	wget https://data.vision.ee.ethz.ch/daid/MOT/pathtrack_release_v1.0.zip
	wget https://favyen.com/files/yt-walking.zip
	mkdir /home/ubuntu/data/yt-walking/
	unzip yt-walking.zip -d /home/ubuntu/data/yt-walking/
	mkdir /home/ubuntu/data/pathtrack/
	unzip pathtrack_release_v1.0.zip
	mv pathtrack_release /home/ubuntu/data/pathtrack/

Extract video frames from YT-Walking mp4 files:

	python scripts/ytw-extract.py /home/ubuntu/data/

Convert MOT17 object detections to uniform JSON format:

	python scripts/mot2json.py /home/ubuntu/data/ train
	python scripts/mot2json.py /home/ubuntu/data/ test

Convert PathTrack object detections to uniform JSON format:

	python scripts/pathtrack.py /home/ubuntu/data/

Normalize MOT17 and PathTrack datasets:

	python scripts/symlink.py mot17 /home/ubuntu/data/
	python scripts/symlink.py pathtrack /home/ubuntu/data/

Pre-process each of the three datasets using `scripts/preprocess-info.py` and `scripts/preprocess-matches.go`.

	python scripts/preprocess-info.py mot17 /home/ubuntu/data/ 8
	python scripts/preprocess-info.py pathtrack /home/ubuntu/data/ 8
	python scripts/preprocess-info.py yt-walking /home/ubuntu/data/ 8
	go run scripts/preprocess-matches.go mot17 /home/ubuntu/data/
	go run scripts/preprocess-matches.go pathtrack /home/ubuntu/data/
	go run scripts/preprocess-matches.go yt-walking /home/ubuntu/data/

Train the model:

	mkdir /home/ubuntu/model/
	python train.py /home/ubuntu/data/ /home/ubuntu/model/model
