from imutils.video import VideoStream
from imutils.video import FPS
import sys
import numpy as np
import time
import cv2

"""
Initial testing for real time cam
"""

# using python 3.6.8
print('Python Version: ' + sys.version)

vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

frame = vs.read()
cv2.imshow("Frame", frame)
cv2.waitKey(0)
