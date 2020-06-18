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


def mask(frame):
    sigma=0.5
    gray_image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    v = np.median(gray_image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(gray_image, lower, upper)
    return edged
def prediction_frame(croppedframe,size):
    croppedframe=cv2.cvtColor(croppedframe,cv2.COLOR_GRAY2BGR)
    croppedframe=cv2.resize(croppedframe,(size,size))
    croppedframe = np.expand_dims(croppedframe, axis=0)
    croppedframe.reshape([1,size,size,3]).astype('float32')/255.0
    return croppedframe
def values( a,  b):
    sum=a+b
    diff= a-b
    product = a*b
    return sum,diff,product
def predictframe():
    FONT_HERSHEY_COMPLEX = 3
    video_capture = cv2.VideoCapture(0)
    framerate = video_capture.get(cv2.CAP_PROP_FPS)
    framecount=0
    flagforprediction=False
    size=128
    left_result=0
    right_result=0
    while(True):
        keypressed = cv2.waitKey(1)
        if keypressed == ord('q'):
            break 
        if keypressed == ord('c'):
            flagforprediction=True
        if keypressed == ord('x'):
            flagforprediction=False

        ret,frame = video_capture.read()
        frame1=frame
        # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        leftframe=frame[100:300,50:250]
        rightframe=frame[100:300,400:600]
        cv2.rectangle(frame, (50, 100),(250, 300), (255, 0, 0), 2)
        cv2.rectangle(frame,(400,100),(600,300),(255,0,0),2)
        if flagforprediction:
            leftframe=mask(leftframe)
            rightframe=mask(rightframe)
            cv2.imshow('Left Frame',leftframe)
            # cv2.resizeWindow('Left Frame', 256, 256)
            cv2.imshow('Right Frame',rightframe)
            # cv2.resizeWindow('Right Frame', 300, 300)
            leftframe=prediction_frame(leftframe,size)
            rightframe=prediction_frame(rightframe,size)
            # if(framecount%(15)==0): # prediction after every 15 frames
            leftdigit=np.argmax(model.predict(leftframe))
            rightdigit=np.argmax(model.predict(rightframe))
            left_result=labels.get(str(leftdigit))
            right_result=labels.get(str(rightdigit))
            print("left value"+str(left_result))
            print("right value"+str(right_result))
            cv2.putText(frame1,"Left value"+ str(left_result), (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            cv2.putText(frame1,"Right value"+ str(right_result), (10, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            # if(int(left_result)==0 || int(right_result)==0):

            framecount+=1
        cv2.imshow('frame',frame1)
        if keypressed == ord('d'):
            flagforprediction=False
            cv2.putText(frame1,"left Value = "+str(left_result), (60, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame1,"Right Value = "+str(right_result), (60, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame1,"Sum = "+str(int(right_result)+int(left_result)), (60, 110),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Total',frame1)
if __name__ == '__main__':
    value=0
    model=load_model('C:/Users/anand/Downloads/thirdtestmodel.h5')
    labels={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
    predictframe()
	