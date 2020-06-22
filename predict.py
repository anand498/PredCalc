import numpy as np
import cv2
from keras.models import load_model,Model
 

def prediction_frame(model,frame):
    frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
    frame=cv2.resize(frame,(128,128))
    frame = np.expand_dims(frame, axis=0)
    frame.reshape([1,128,128,3]).astype('float32')/255.0
    value=str(np.argmax(model.predict(frame)))
    return value
def binmask(frame):
        gray = frame[:, :, 2]
        ret, thresh_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
        thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
        return thresh_gray
def calc(a):
    op=['+','-','*','/']
    operand=''
    for i in range(len(a)):
        if(a[i] in op ):
            value1=int(a[0:i])
            value2=int(a[i+1:len(a)-1])
            operand=a[i]
         
        elif (a[i]=='='):
            if(operand=='+'):
                result = value1+value2
            elif (operand=='-'):
                result=value1-value2
            elif (operand=='*'):
                result=value1*value2        
    return str(result)
def hand_calc():
    framecount=0
    dig_pred=False
    text=""
    oldtext=""
    digit=""
    sametext=0
    equal=True
    totalframecheck=40
    while(True):
        display = np.zeros((480, 600, 3), dtype=np.uint8)
        keypressed = cv2.waitKey(1)
        if keypressed == ord('q'): # to exit
            break 
        if keypressed == ord('c'): # to start the prediction
            dig_pred=True
        if keypressed == ord('x'): # to close the masked frame
            dig_pred=False
        if keypressed == ord('o'):
            cv2.destroyWindow('Cropped Frame')
        ret,frame = video_capture.read()
        digitframe=frame[113:263,23:173]
        digitframe=binmask(digitframe)
        cv2.putText(frame,"Digit Window",(10, 90),font,fontsize,red,thickness)
        cv2.imshow('Digit Mask ',digitframe)
        cv2.rectangle(frame, (21, 111),(175, 265),blue, 1)
        if dig_pred:
            oldtext=digit
            digit=prediction_frame(maskmodel,digitframe)
            if oldtext==digit:
                sametext+=1 
            else:
                sametext=0
            if sametext==totalframecheck and digit=='0':
                text=''
                sametext=0
            if sametext==totalframecheck:
                if(digit!='13'):
                    text+=labels.get(digit)
                    sametext=0
                if digit=='13':
                    text+='='
                    text+=calc(text)
                    sametext=0
        
        framecount+=1
        cv2.putText(display,"Equation: "+ text, (10, 150),font,fontsize+.6,green,thickness) 
        cv2.putText(display,"Predicted Value: "+str(labels.get(digit)), (10, 80),font,fontsize,red,thickness)
        frame=np.hstack((frame,display))
        cv2.imshow('frame',frame)
        
if __name__ == '__main__':
    (red,green,blue)=((0,0,255),(0,255,0),(255,0,0))
    font=cv2.FONT_HERSHEY_TRIPLEX
    thickness=1
    fontsize=0.6
    text=""
    maskmodel=load_model('C:/Users/anand/Downloads/maskmodelv1.h5')
    # '0':'','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','10':'add','11':'sub','12':'multiply','13':'divide'
    labels={'0':'Empty','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','10':'+','11':'-','12':'*','13':'='}
    video_capture = cv2.VideoCapture(0)
    hand_calc()
	