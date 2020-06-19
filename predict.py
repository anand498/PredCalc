import numpy as np
import cv2,os,sys,dlib,imutils
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

def prediction_frame(frame,size):
    frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
    frame=cv2.resize(frame,(size,size))
    frame = np.expand_dims(frame, axis=0)
    frame.reshape([1,size,size,3]).astype('float32')/255.0
    return frame

def values( a,  b):
    a=int(a)
    b=int(b)
    sum=a+b
    diff= a-b 
    product = a*b
    return sum,diff,product

def predictframe():
    video_capture = cv2.VideoCapture(0)
    framecount=0
    size=128
    flagforprediction=False
    arr=[]
    thickness=1
    fontsize=0.6
    (red,green,blue)=((0,0,255),(0,255,0),(255,0,0))
    font=cv2.FONT_HERSHEY_COMPLEX
    display = np.zeros((480, 480, 3), dtype=np.uint8)
	
    while(True):
        keypressed = cv2.waitKey(1)
        if keypressed == ord('q'):
            break 
        if keypressed == ord('c'):
            flagforprediction=True
        if keypressed == ord('x'):
            flagforprediction=False
            cv2.destroyWindow('Cropped Frame')
        ret,frame = video_capture.read()
        frame1=frame
        frame2=frame
        digitframe=frame[100:300,400:600]
        operatorframe=frame[100:300,10:210]
        cv2.putText(display,"Digit Window ", (400, 90),font,fontsize,green,thickness)
        cv2.putText(frame1,"Operator Window ", (10, 90),font,fontsize,green,thickness)
        cv2.rectangle(frame, (10, 100),(210, 300),blue, 1)
        cv2.rectangle(frame,(400,100),(600,300),blue,1)
        if flagforprediction:
            digitframe=mask(digitframe)
            operatorframe=mask(operatorframe)
            cv2.imshow('Operator',operatorframe)
            cv2.imshow('Digit ',digitframe)
            digitframe=prediction_frame(digitframe,size)
            operatorframe=prediction_frame(operatorframe,size)
            if(framecount%4==0):
                digit=str(np.argmax(model.predict(digitframe)))
                operator=str(np.argmax(operatormodel.predict(operatorframe)))
                # operator=labels.get(str(operator))
                arr.append(digit)
            if(digit!='0'):
                cv2.putText(display," Number"+ digit, (10, 40),font,fontsize,red,thickness)
                cv2.putText(display," Operator"+ operator, (200, 40),font,fontsize,red,thickness)
                
            if(digit=='0'):
                # flagforprediction=False
                prevright=arr[len(arr)-2]
                cv2.putText(display,"Previous Value "+ prevright, (120, 40),font,fontsize,green,thickness)
            framecount+=1
            frame=np.hstack((frame,display))
        cv2.imshow('frame',frame)
if __name__ == '__main__':
    model=load_model('C:/Users/anand/Downloads/thirdtestmodel.h5')
    operatormodel=load_model('C:/Users/anand/Downloads/secondoperatormodel.h5')
    labels={'0':'add','1':'cool','2':'palm','3':'yo','4':'next'}
    predictframe()
	