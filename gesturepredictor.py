import numpy as np
import cv2
import os 
import shutil
try:
	os.mkdir('masks')# make a directory contatining the masks
except:
	shutil.rmtree('masks',ignore_errors=True)# if the masks folder exists
	os.mkdir('masks') #create a new directory
def binmask(frame):
		gray = frame[:, :, 2]
		ret, thresh_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
		thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
		return thresh_gray
def captureframe(value,totalframes):
	video_capture = cv2.VideoCapture(0)
	framecount,framenum,flagforprediction=(0,0,False)
	while(True):
		keypress= cv2.waitKey(1)
		ret,frame = video_capture.read()
		if  keypress == ord('q'):
    			break
		if  keypress ==  ord('c'):
    			flagforprediction = True		
		framecount+=1
		cv2.rectangle(frame, (21, 111),(175, 265), (255, 0, 0), 1)# create a window for capturing masks
		if flagforprediction:
				cv2.putText(frame,'Capturing Frame: ', (15, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				if(framenum!=totalframes and framecount%2==0): # every 2nd frame
						cv2.imwrite('masks/%d-%d.png'%(value,framenum),binmask(frame[113:263,23:173]))
						print('Frame Captured: '+str(framenum))
						framenum+=1
						if (framenum==totalframes):
								print('Finished Capturing')
								flagforprediction= False
								break
		cv2.imshow('Actual Frame',frame)
		cv2.imshow('Mask Frame',binmask(frame))

if __name__ == '__main__':
	print('Enter the Gesture Label in Numeric')
	a=int(input()) 
	print('Enter the total number of frames')
	numofframes=int(input())
	captureframe(a,numofframes)
	