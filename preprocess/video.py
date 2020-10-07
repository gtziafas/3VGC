import cv2
import numpy as np

from typing import Optional, Tuple


class VideoBuffer(object):
    def __init__(self, path: str) -> None:
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.nframes / self.fps
        self.buffer = self.get_buffer()

    def get_buffer(self) -> np.ndarray:
        buffer = np.empty((self.nframes, self.height, self.width, 3), dtype='uint8')
        ret, f = True, 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            buffer[f] = frame 
            f += 1
        # re-init cap
        self.cap.release()
        self.cap = cv2.VideoCapture(self.path)
        return buffer

    def play(self, 
            buffer: Optional[np.ndarray]=None, 
            fps: Optional[float]=None
        ) -> None:
        buffer = self.buffer if buffer is None else buffer
        fps = self.fps if fps is None else fps 
        delay_in_ms = int(1e03 / int(fps))
        for f in range(buffer.shape[0]):
            frame = buffer[f]
            cv2.imshow(self.path, frame)
            key = cv2.waitKey(delay_in_ms) & 0xff
            if key == ord('q'):
                break
        cv2.destroyWindow(self.path)

    def resize(self, 
            size: Tuple[int, int],
            buffer: Optional[np.ndarray]=None
        ) -> np.ndarray:
        buffer = self.buffer if buffer is None else buffer
        h, w = size
        resized = np.empty((buffer.shape[0], h, w, 3), dtype='uint8')
        maxdim, mindim = max(self.height, self.width), min(self.height, self.width)
        for f in range(buffer.shape[0]):
            frame = buffer[f]
            q = np.zeros((maxdim, maxdim, 3), dtype='uint8')
            start = int(0.5 * (maxdim-mindim))
            end = int(0.5 * (maxdim+mindim))
            if mindim == frame.shape[0]:
                q[start:end, :, :] = frame 
            elif mindim == frame.shape[1]:
                q[:, start:end, :] = frame 
            resized[f] = cv2.resize(q, (h, w))
        return resized 

    def subsample(self, 
                new_fps: int,
                old_fps: Optional[float]=None, 
                buffer: Optional[np.ndarray]=None
    )-> np.ndarray:
        buffer = self.buffer if buffer is None else buffer
        old_fps = self.fps if old_fps is None else old_fps
        step = round(old_fps / new_fps)
        return buffer[0::step, ...]


import torch
import torch.nn.functional as F
class Test(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(3, 16, (2,3,3))
        self.conv2 = torch.nn.Conv3d(16, 32, (2,3,3))
        self.conv3  = torch.nn.Conv3d(32, 64, (2,3,3))
        self.conv4 = torch.nn.Conv3d(64, 64, (1,3,3))
        self.classifier = torch.nn.Linear(576, 8)

    def forward(self, x):
        x= self.conv1(x)
        x= F.max_pool3d(x, (2,3,3))
        print(x.shape)
        x= self.conv2(x)
        x= F.max_pool3d(x, (2,3,3))
        print(x.shape)
        x= self.conv3(x)
        x= F.max_pool3d(x, (2,3,3))
        print(x.shape)
        x= self.conv4(x)
        print(x.shape)
        x= x.flatten(1)
        return self.classifier(x)

