# HandWritten_MathematicalSymbol_Classification
# Final Project

Classify the images of handwritten mathematical symbols. 

## Code Implementation & Technical Report

Every group was given the same data sets: "data_train.npy" and "t_train.npy".
Our group went through and cleaned and relabeled this data. See Homework 3 Part 2 file: "Homework 3 Part 2 -- Data Cleaned.ipynb". 

## Training Data

We trained the model using the VGG16 (type of convolutional neural network) algorithm. The training and prepocessing was done in the "train.ipynb" file. The model with the optimal parameters was saved and exported to "trained_VGG16_model.h5". The test program uses this exported model.
Below are the model parameters:  
<img src="https://github.com/AmitKumar7138/HandWritten_MathematicalSymbol_Classification/blob/main/model_params.jpg" width="500" height="800">

The model was trained on corrected label and corrected image dataset.

Train function takes in the image_data.npy file and labels.npy file to train the model and stores the "trained_VGG16_model.h5" in the same directory. It also returns a history callback object for further analysis along with a detailed plot of accuracy and loss.

## Testing

test.ipynb consists the test function for the easy dataset which will be in the same format as the training dataset provided to us initially. The test function takes in image_data.npy and label.npy. It loads the "trained_VGG16_model.h5" from the same directory to run the predictions and gives out predicted labels in integer encoding along with the accuracy score. 

IMPORTANT: It is trained to detect the unknown character so the label "10" will be produced from the image that does not belong to any of the classes listed in the table given in instructions. 

## Dependencies

- matplotlib==3.5.3
- numpy==1.23.4
- pandas==1.5.1
- sklearn==1.1.2
- tensorflow==2.6.0
- keras==2.6.0
- cv2==4.5.1

## Authors

Archit Jaiswal, Amit Kumar, Nathan Grinalds, Humberto Garcia

## Link: https://github.com/UF-EEL5840-F22/final-project---code-report-sosimple
