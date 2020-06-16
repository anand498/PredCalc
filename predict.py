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
def prediction_frame(croppedframe):
    croppedframe=cv2.cvtColor(croppedframe,cv2.COLOR_GRAY2BGR)
    croppedframe=cv2.resize(croppedframe,(28,28))
    croppedframe = np.expand_dims(croppedframe, axis=0)
    croppedframe.reshape([1,28,28,3])
    croppedframe = croppedframe.astype('float32')
    croppedframe = croppedframe / 255.0
    return cropped frame

def predictframe():
    FONT_HERSHEY_COMPLEX = 3
    video_capture = cv2.VideoCapture(0)
    framerate = video_capture.get(cv2.CAP_PROP_FPS)
    while(True):
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break 
        ret,frame = video_capture.read()
        frame1=frame
        leftframe=frame[100:300,0:200]
        rightframe=frame[100:300,500;700]
        cv2.rectangle(frame, (0, 100),(200, 300), (255, 0, 0), 2)
        cv2.rectangle(frame,(500,100),(700,300),(255,0,0),2)
        leftframe=mask(leftframe)
        righ
        cv2.imshow('Masked',croppedframe)
        a=np.argmax(model.predict(croppedframe))+1
        result=labels.get(str(a))
        cv2.putText(frame1,str(result), (10, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Frame',frame1)
if __name__ == '__main__':
	model=load_model('C:/Users/anand/Desktop/firstfmodel.h5')
    labels={'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':'minus','11':'multiply','12':'addition','13':'fist','14':'palm'}
	predictframe()
	