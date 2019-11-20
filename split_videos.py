import cv2
from pathlib import Path
import os
import numpy as np 

video_dir = './'
fps = 60
OUT_VIDEO_WRITE_FILE = 'bb.avi'

class Video():
	def __init__(self, from_path=None):
		self.filename = self.get_filename(from_path)
		self.cap = cv2.VideoCapture(str(from_path))
		self.cap.set(cv2.CAP_PROP_FPS, 10)
		self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
		self.fps = int(self.cap.get(fps))
		#self.out = cv2.VideoWriter(OUT_VIDEO_WRITE_FILE, self.fourcc, self.fps, (640,480))
		
	@staticmethod
	def get_filename(file):
		file = file.split('.')[0]

		return file

	def split(self, idx):
		# name the destination file name
		out_name = self.filename + '_' + str(idx) + '.avi'

		# start a new Video Saver object
		out = cv2.VideoWriter(out_name, self.fourcc, self.fps, (480,360))
		#self.delay_in_ms = 1/self.fps * 1e03
		count = 0 
		while (self.cap.isOpened()):
			# take the frame
			ret, frame = self.cap.read()
			if not ret:
				break

			# save the frame 
			out.write(frame)
			key = cv2.waitKey(1) & 0xFF


		self.cap.release()
		out.release()

def test():
	#pathlist = Path(video_dir).glob('**/*.mp4')
	#count = 0
	#for path in pathlist:
	file = str('test.mp4')
	print(file)
	v = Video(from_path = file)
	v.split(0)
		#cap = cv2.VideoCapture(file)
		#while cap.isOpened():
		#	ret, frame = cap.read()
		#	cv2.imshow('f',frame)
		#	key = cv2.waitKey(1) & 0xFF

			# check for keyboard interaction
		#	if key == ord('q'):		# q is for quitting
		#		break

		#count += 1

def test2():

	cap = cv2.VideoCapture('test.mp4')

	frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

	fc = 0
	ret = True

	while (fc < frameCount  and ret):
	    ret, buf[fc] = cap.read()
	    fc += 1

	cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
	duration = cap.get(cv2.CAP_PROP_POS_MSEC)
	cap.release()
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(OUT_VIDEO_WRITE_FILE, fourcc, 25, (buf.shape[2],buf.shape[1]))
	
	#cv2.namedWindow('frame 10')
	#cv2.imshow('frame 10', buf[9])
	#print(buf.shape)
	bb = buf[0::int(buf.shape[0]/25)]
	#print(bb.shape)
	#cv2.waitKey(0)

	for _ in range(bb.shape[0]):
		out.write(bb[_,:,:,:])

	out.release()

test2()