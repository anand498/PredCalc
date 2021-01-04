# PredCalc
An application that performs calculations based on the gestures. With the help of Deep learning and Computer Vision, the system 
identifies the gesture signal and predict the combined expression intended.The system detects hand gestures and perfrom the calculations accordingly.<br />
The labels are '0'-for clear screen ,1-9 - American Sign Language,10-addition,11-subtraction,12-multiply,13-division,14-get the result of expression. To further inderstand the labels I have used, go to the `labels` folder.


## File Specifications
[requirements.txt](https://github.com/anand498/PredCalc/blob/master/requirements.txt):
Containing the libraries and modules required to run the project.

[gesturepredictor.py](https://github.com/anand498/PredCalc/blob/master/gesturepredictor.py):
This script is used to capture gestures for creating a dataset for having any number of labels and gestures you want.

[gesture-training.py](https://github.com/anand498/PredCalc/blob/master/gesture-training.py):
This script will be used to make a numpy array of the model and train the dataset for creating model for prediction.

[predict.py](https://github.com/anand498/PredCalc/blob/master/predict.py) :
The main script that loads the trained model and performs the prediction of the gestures indicated by the user and the calculation of the total expression.

To download the required libraries: <br />
`pip install -r requirements.txt ` <br />

Download the model from this [link](https://drive.google.com/file/d/1EdlSt_bHTxw1wW68AoLlZtvF3rvN3Bjl/view?usp=sharing) and store it in this directory

### Make your own dataset: 
`python facepredictor.py` <br />
 Enter the label for the mask <br />
 Enter the total frames

### Train the dataset with a CNN model
`python gesture-training.py` <br/>
The outcome of the file would be a model (.h5) to predict the new gestures made by the user for performing the calculations in run-time.

### Here's the Drive link to the pre-trained model
https://drive.google.com/file/d/1R1CSScF4EdaDnyhBJC2wpZTM9lZVmpEg/view?usp=sharing

### Predicting the gestures in real-time
`python predict.py` <br />
After you esecture the command place your hand in the region inside the blue window and press the 'c' key once. The model will start predicting the gestures in that particular frame window. Now to confirm the prediction you have to maintain the same frame for atleast 40 frames. After that the value gets appended into the expression. Once you have correctly formed the expression tehe pointing out your little finger(label 14) the result of this expression gets appended into the expression. To clear the screen, show a pumped fist(label 0)
 Press 'c' for starting the prediction<br />
 Press 'q' to stop the application<br />
 Press 'x' for pausing the prediction<br />



