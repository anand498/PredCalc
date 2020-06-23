import numpy as np
import cv2
from keras.models import load_model
 

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
def hand_calc():
    framecount=0
    dig_pred=False
    text,oldtext,digit=('','','')
    # text=""
    # oldtext=""
    # digit=""
    sametext=0
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
        digitframe=frame[113:263,23:173] #cropped frame
        digitframe=binmask(digitframe)
        cv2.putText(frame,"Digit Window",(10, 90),font,fontsize,red,thickness) #display the text digit window
        cv2.imshow('Digit Mask ',digitframe)
        cv2.rectangle(frame, (21, 111),(175, 265),blue, 1) #create a rectangle for the region of interet segmentation
        if dig_pred:
            oldtext=digit
            digit=prediction_frame(maskmodel,digitframe) # predict the segmented ROI
            if oldtext==digit: #check if values are the same
                sametext+=1 
            else:
                sametext=0
            if sametext==totalframecheck and digit=='0': # to clear the screen
                text=''
                sametext=0
            if sametext==totalframecheck:# if the value is detected for a ceratin number of consecutive frames
                if(digit!='13'):# to check for the digit it should not be the '=' sign
                    text+=labels.get(digit)
                    sametext=0
                if digit=='13':
                    print(text[:len(text)-1])
                    text+=str(eval(text[:len(text)-1]))
                    sametext=0
        
        framecount+=1
        cv2.putText(display,"Equation: "+ text, (10, 150),font,fontsize+.6,green,thickness) 
        cv2.putText(display,"Predicted Value: "+str(labels.get(digit)), (10, 80),font,fontsize,red,thickness)
        frame=np.hstack((frame,display))
        cv2.imshow('frame',frame)
        
if __name__ == '__main__':
    (red,green,blue)=((0,0,255),(0,255,0),(255,0,0))
    font,thickness,fontsize=(cv2.FONT_HERSHEY_TRIPLEX,1,0.6)
    # font=cv2.FONT_HERSHEY_TRIPLEX
    # thickness=1
    # fontsize=0.6
    maskmodel=load_model('C:/Users/anand/Downloads/maskmodelv1.h5')
    # '0':'','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','10':'add','11':'sub','12':'multiply','13':'divide'
    labels={'0':'Empty','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','10':'+','11':'-','12':'*','13':'='}
    video_capture = cv2.VideoCapture(0)
    hand_calc()
	