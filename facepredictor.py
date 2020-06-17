from keras.preprocessing.image import img_to_array
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import os
import sys
import dlib
from PIL import Image
from keras.models import load_model,Model
from keras.models import Model,load_model
from keras.preprocessing.image import img_to_array
import numpy as np # linear algebra

def mask(frame):
		lower_red = np.array([30,150,50]) 
		upper_red = np.array([255,255,180]) 
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv, lower_red, upper_red) 
		res = cv2.bitwise_and(frame,frame, mask= mask) 
		edges = cv2.Canny(frame,100,200) 
		return edges
    	
def captureframe():
    	
	video_capture = cv2.VideoCapture(0)
	framerate = video_capture.get(cv2.CAP_PROP_FPS)
	print(framerate)
	framecount = 0
	digitcount=0
	while(True):
		if cv2.waitKey(1) & 0xFF == ord('q'):
					break 
		ret,frame = video_capture.read()
		framecount+=1
		# cv2.imshow("Frame",frame)
		cv2.rectangle(frame, (5, 95),(215, 310), (255, 0, 0), 1)
		cv2.imshow('Frame',frame)
		if(framecount%(framerate/3)==0 and digitcount!=60):
				print('Captured :'+str(digitcount))
				digitcount+=1
				cv2.imwrite('C:/Users/anand/Desktop/data/09_%d.png'%np.random.randint(1,40000),frame[100:300,10:210])
				if (digitcount==60):
    					break
if __name__ == '__main__':
	captureframe()
	