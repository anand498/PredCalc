from keras.preprocessing.image import img_to_array
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import numpy as np
import cv2,os,sys,dlib,imutils
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
    # croppedframe=cv2.cvtColor(croppedframe,cv2.COLOR_GRAY2BGR)
    croppedframe=cv2.resize(croppedframe,(128,128))
    croppedframe = np.expand_dims(croppedframe, axis=0)
    croppedframe.reshape([1,128,128,3]).astype('float32')/255.0
    # croppedframe = croppedframe.astype('float32')
    # croppedframe = croppedframe / 255.0
    return croppedframe

def predictframe():
    FONT_HERSHEY_COMPLEX = 3
    video_capture = cv2.VideoCapture(0)
    framerate = video_capture.get(cv2.CAP_PROP_FPS)
    framecount=0
    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
        ret,frame = video_capture.read()
        frame1=frame
        leftframe=frame[100:300,50:250]
        rightframe=frame[100:300,400:600]
        
        cv2.rectangle(frame, (50, 100),(250, 300), (255, 0, 0), 2)
        cv2.rectangle(frame,(400,100),(600,300),(255,0,0),2)
        cv2.imshow('frame',rightframe)

        # leftframe=mask(leftframe)
        # rightframe=mask(rightframe)
        # cv2.imshow('Left Frame',leftframe)
        # cv2.resizeWindow('Left Frame', 256, 256)
        # cv2.imshow('Right Frame',rightframe)
        # cv2.resizeWindow('Right Frame', 300, 300)
        leftframe=prediction_frame(leftframe)
        rightframe=prediction_frame(rightframe)
        # if(framecount%(15)==0): # prediction after every 15 frames
        leftdigit=np.argmax(model.predict(leftframe))+1
        rightdigit=np.argmax(model.predict(rightframe))+1
        print(rightdigit)
        left_result=labels.get(str(leftdigit))
        right_result=labels.get(str(rightdigit))
        cv2.putText(frame1,str(left_result), (10, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame1,str(right_result), (30, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
        cv2.imshow('Frame',frame1)
        framecount+=1
if __name__ == '__main__':
    model=load_model('C:/Users/anand/Desktop/firstfmodel.h5')
    labels={'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
    predictframe()
	