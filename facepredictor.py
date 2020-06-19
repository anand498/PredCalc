import numpy as np
import imutils
import cv2
import os
import sys
import dlib
from keras.models import load_model,Model
from keras.preprocessing.image import img_to_array


def mask(frame):
		sigma=0.5
		gray_image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		v = np.median(gray_image)
		lower = int(max(0, (1.0 - sigma) * v))
		upper = int(min(255, (1.0 + sigma) * v))
		edged = cv2.Canny(gray_image, lower, upper)
		return edged
def captureframe():
	video_capture = cv2.VideoCapture(0)
	framerate = video_capture.get(cv2.CAP_PROP_FPS)
	print(framerate)
	framecount = 0
	flagforprediction = False
	digitcount=0
	while(True):
		keypress= cv2.waitKey(1)
		if  keypress == ord('q'):
					break 
		if  keypress ==  ord('c'):
    			flagforprediction = True		
		ret,frame = video_capture.read()
		frame1=frame
		framecount+=1
		totalframes=200
		cv2.imshow('Masked',mask(frame1))
		cv2.rectangle(frame, (23, 113),(237, 352), (255, 0, 0), 1)
		if flagforprediction==True:
				if(framecount%3==0 and digitcount!=totalframes):
						print('Captured : '+str(digitcount))
						digitcount+=1
						cv2.putText(frame,'Capturing', (300, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
						# cv2.imwrite('C:/Users/anand/Desktop/Data/9-%d.png'%np.random.randint(1,40000),mask(frame[115:350,25:235]))
						if (digitcount==totalframes):
							flagforprediction= False
		cv2.imshow('Actual Frame',frame)
		cv2.imshow('Masked frame',mask(frame))		
if __name__ == '__main__':
	captureframe()
	