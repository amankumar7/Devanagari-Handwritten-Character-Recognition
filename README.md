# Devanagari Handwritten Character Recognition

### Discription
A Sequential Convolution Neural Network is used for recogination. The data set contain 46 classes which includes numeric and alphabets of size 32 * 32 pixels.


#### [Download database](https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset) 

##### Train Accuracy - 85 ~ 90
##### Test Accuracy - 94.14 (Highest)

### Procedure:

##### 1. Preprocessing

```
python data_preprocessing.py
```

1) Load data using OpenCV and convert each images into grayscale.
2) Resize the images and append to a single array .
3) Construct a dictionary which will map the labels which we extracted from the folder name for each classes.
4) Shuffle the data and return.


##### 2. Training and Evaluation 

```
python training.py
```

1) Load the pickle files for training.
2) Reshape the data as needed by keras and convert the lables into one hot array.
3) Normalize the data .
4) Define the architecture of CovNet.
5) Use 20% data for validation.
6) Train the model

##### Architecture 

##### Con2D => Con2d => Relu => MaxPooling => Con2D => Relu => MaxPooling => Dense => Dropout => Output 

##### Activation fuctions : Rectify Linear Unit and Softmax(output)

##### Optimizer : Adam
