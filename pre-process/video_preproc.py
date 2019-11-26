import cv2

class Video():
	def __init__(self, path: str) -> None:
		self.path = path 
		self.cap = cv2.VideoCapture(self.path)
		self.out = cv2.VideoWriter(OUT_VIDEO_WRITE_FILE, self.fourcc, self.fps, (640,480))