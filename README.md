# Classification-of-Dogs-and-Cats-using-a-CNN
This project demonstrates how to use a convolutional neural network (CNN) to classify images of dogs and cats. The model is trained on a dataset of images of dogs and cats, and is able to predict whether an image contains a dog or a cat with high accuracy.

## Requirements
To run this project, you will need the following software and libraries:

Python 3.x
TensorFlow
NumPy
OpenCV

## Usage
To run the project, simply execute the main.py script. The script will automatically load and preprocess the data, define the model architecture, and train the model. The trained model will then be evaluated on the test data, and the accuracy will be printed to the console.

## Data
The data for this project is a subset of the Dogs vs. Cats dataset from Kaggle. The data consists of 25,000 images of dogs and cats, evenly split between the two classes. The images have been resized to 150x150 pixels and preprocessed for use with the CNN.

## Model
The model for this project is a simple CNN consisting of three convolutional layers, each followed by a max pooling layer. The output of the final pooling layer is flattened and passed through a fully-connected layer with a single output node, which is used to predict the class of the input image. The model is trained using the Adam optimizer and the binary cross-entropy loss function.

##Results
On the test data, the model is able to achieve an accuracy of approximately 85%. This demonstrates the effectiveness of CNNs for image classification tasks, and shows the potential for using this model for real-world applications such as automatically classifying images of animals in wildlife photography or assisting with animal identification in research.
