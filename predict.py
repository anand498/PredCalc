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

def prediction_frame(frame):
    frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
    frame=cv2.resize(frame,(128,128))
    frame = np.expand_dims(frame, axis=0)
    frame.reshape([1,128,128,3]).astype('float32')/255.0
    digit=str(np.argmax(model.predict(frame)))
    operator=str(np.argmax(operatormodel.predict(frame)))
    return digit,operator


def predictframe():
    framecount,newdigit,size=(0,0,128)
    digitprediction=(False)
    arr=[]
    text=""
    oldtext=""
    digittodisplay=[]
    while(True):
        display = np.zeros((480, 480, 3), dtype=np.uint8)
        keypressed = cv2.waitKey(1)
        if keypressed == ord('q'): # to exit
            break 
        if keypressed == ord('c'): # to start the prediction
            digitprediction=True
        if keypressed == ord('x'): # to close the masked frame
            digitprediction=False
        if keypressed == ord('o'):
            cv2.destroyWindow('Cropped Frame')
        ret,frame = video_capture.read()
        digitframe=frame[100:300,10:210]
        digitframe=mask(digitframe)
        cv2.putText(frame,"Digit Window",(10, 90),font,fontsize,green,thickness)
        cv2.imshow('Digit ',digitframe)
        cv2.rectangle(frame, (10, 100),(210, 300),blue, 1)

        if digitprediction:
            digit,operator=prediction_frame(digitframe) 
            digittodisplay.append(digit)
            print(digittodisplay)

            # if(len(digittodisplay))>=20:
                # for i in range(len(digittodisplay)-1,len(digittodisplay)-20,-1):                        
            #         if(digittodisplay[i]==digittodisplay[i-1]):
            #             newdigit=digittodisplay[len(digittodisplay)-1]
            #             if newdigit=='0' and digittodisplay[len[digittodisplay]-2!='0']:
            #                 text+=str(newdigit)
        framecount+=1
        # if(str(newdigit)=='0'):
            # text+=str(newdigit)
        cv2.putText(display, str(text), (30, 40),font,fontsize,green,thickness)
        cv2.putText(display,str(newdigit), (100, 40),font,fontsize,red,thickness) 
        cv2.putText(display," Operator: ", (200, 40),font,fontsize,red,thickness)
        frame=np.hstack((frame,display))
        cv2.imshow('frame',frame)
if __name__ == '__main__':
    (red,green,blue)=((0,0,255),(0,255,0),(255,0,0))
    font=cv2.FONT_HERSHEY_TRIPLEX
    thickness=1
    fontsize=0.6
    text=""
    model=load_model('C:/Users/anand/Downloads/thirdtestmodel.h5')
    operatormodel=load_model('C:/Users/anand/Downloads/secondoperatormodel.h5')
    labels={'0':'add','1':'cool','2':'palm','3':'yo','4':'next'}
    video_capture = cv2.VideoCapture(0)
    predictframe()
	