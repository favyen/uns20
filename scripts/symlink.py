import os
import subprocess
import sys

dataset = sys.argv[1]
data_path = sys.argv[2]

if dataset == 'mot17':
    os.makedirs(data_path + '/mot17/json', exist_ok=True)
    os.makedirs(data_path + '/mot17/frames', exist_ok=True)
    for split in ['train', 'test']:
        labels = os.listdir(data_path + 'mot17/{}'.format(split))
        labels = [label for label in labels if 'SDP' in label]
        for label in labels:
            subprocess.call(['ln', '-s', data_path + '/mot17/{}/{}/det/det-filter60.json'.format(split, label), data_path + '/mot17/json/{}.json'.format(label)])
            subprocess.call(['ln', '-s', data_path + '/mot17/{}/{}/img1'.format(split, label), data_path + '/mot17/frames/{}'.format(label)])
elif dataset == 'pathtrack':
    os.makedirs(data_path + '/pathtrack/frames', exist_ok=True)
    labels = os.listdir(data_path + 'pathtrack/train')
    for label in labels:
        subprocess.call(['ln', '-s', data_path + '/pathtrack/train/{}/img1'.format(label), data_path + '/pathtrack/frames/{}'.format(label)])
