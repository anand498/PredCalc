# PredCalc
An application that performs calculations based on the gestures. With the help of Deep learning and Computer Vision, the system
identifies the gesture signal and predict the combined expression intended.

### About the System



## File Specifications
[requirements.txt](https://github.com/anand498/PredCalc/blob/master/requirements.txt):
Containing the libraries and modules required to run the project.

[gesturepredictor.py ](https://github.com/anand498/PredCalc/blob/master/gesturepredictor.py):
This script is used to capture gestures for creating a dataset for having any number of labels and gestures you want.

[gesture-training.py](https://github.com/anand498/PredCalc/blob/master/gesture-training.py):
This script will be used to make a numpy array of the model and train the dataset for creating model for prediction

[predict.py](https://github.com/anand498/PredCalc/blob/master/predict.py) :
The main script that loads the trained model and performs the prediction of the gestures indiacted by the user and the calculation of the total expression


## Requirements
* Python3
* Tensorflow
* Keras
* numpy
* Imutils
* OpenCV (cv) for python3

To download the required libraries:
> pip install -r requirements.txt 

### Make your own dataset:
> python facepredictor.py
 Enter the label for the mask
 Enter the total frames

### Train the dataset with a CNN model
> python gesture-training.py
The outcome of the file would be a model (.h5) to predict the new gestures made by the user for performing the calculations in run-time.

### Predicting the gestures in real-time
> python predict.py
After you esecture the command place your hand in the region inside the blue window and press the 'c' key once. The model will start predicting the gestures in that particular frame window.
Now to confir




