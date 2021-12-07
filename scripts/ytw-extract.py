import os
import subprocess
import sys

data_path = sys.argv[1]

processes = []
for fname in os.listdir(data_path+'/yt-walking/'):
	if not fname.endswith('.mp4'):
		continue
	label = fname.split('.')[0]
	os.makedirs(data_path+'/yt-walking/frames/' + label, exist_ok=True)
	p = subprocess.Popen(['ffmpeg', '-threads', '3', '-i', data_path+'/yt-walking/'+fname, '-vf', 'fps=10,scale=960:540', '-q:v', '1', data_path+'/yt-walking/frames/' + label + '/%06d.jpg'])
	processes.append(p)
for p in processes:
	p.wait()
