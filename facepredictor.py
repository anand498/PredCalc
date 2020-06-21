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
def binmask(frame):
		gray = frame[:, :, 2]
		ret, thresh_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
		thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
		return thresh_gray
def captureframe(value):
	video_capture = cv2.VideoCapture(0)
	framecount = 0
	flagforprediction = False
	digitcount=0
	framenum=0
	while(True):
		keypress= cv2.waitKey(1)
		ret,frame = video_capture.read()
		if  keypress == ord('q'):
    			break
		if  keypress ==  ord('c'):
    			flagforprediction = True		
		framecount+=1
		totalframes=500
		cv2.rectangle(frame, (21, 111),(175, 265), (255, 0, 0), 1)
		if flagforprediction==True:
				if(digitcount!=totalframes and framecount%2==0):
						print('Captured : '+str(digitcount))
						digitcount+=1
						cv2.putText(frame,'Capturing', (300, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
						cv2.imwrite('C:/Users/anand/Desktop/masks/%d-%d.png'%(int(value),framenum),binmask(frame[113:263,23:173]))
						framenum+=1
						if (digitcount==totalframes):
							flagforprediction= False
							break
		cv2.imshow('Actual Frame',frame)
		cv2.imshow('Mask Frame',binmask(frame))

if __name__ == '__main__':
	print('Enter the value')
	a=input()
	captureframe(a)
	