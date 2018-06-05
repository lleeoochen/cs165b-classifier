# Starter code for CS 165B HW4

"""
Implement the testing procedure here. 

Inputs:
    Given the folder named "hw4_test" that is put in the same directory of your "predictio.py" file, like:
    - Main folder
        - "prediction.py"
        - folder named "hw4_test" (the exactly same as the uncompressed hw4_test folder in Piazza)
    Your "prediction.py" need to give the following required output.

Outputs:
    A file named "prediction.txt":
        * The prediction file must have 10000 lines because the testing dataset has 10000 testing images.
        * Each line is an integer prediction label (0 - 9) for the corresponding testing image.
        * The prediction results must follow the same order of the names of testing images (0.png - 9999.png).
    Notes: 
        1. The teaching staff will run your "prediction.py" to obtain your "prediction.txt" after the competition ends.
        2. The output "prediction.txt" must be the same as the final version you submitted to the CodaLab, 
        elsewise you will be given 0 score for your hw4.


**!!!!!!!!!!Important Notes!!!!!!!!!!**
    To open the folder "hw4_test" or load other related files, 
    please use open('./necessary.file') instaed of open('some/randomly/local/directory/necessary.file').

    For instance, in the student Jupyter's local computer, he stores the source code like:
    - /Jupyter/Desktop/cs165B/hw4/prediction.py
    - /Jupyter/Desktop/cs165B/hw4/hw4_test
    If he use os.chdir('/Jupyter/Desktop/cs165B/hw4/hw4_test'), this will cause an IO error 
    when the teaching staff run his code under other system environments.
    Instead, he should use os.chdir('./hw4_test').


    If you use your local directory, your code will report an IO error when the teaching staff run your code,
    which will cause 0 socre for your hw4.
"""


import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import sys
import os


# Constants
test_path = './hw4_test/'
train_path = './hw4_train/'
result_file = 'prediction.txt'
model_file = 'model.tflearn'


# Print current load status.
def printStatus(status):
    sys.stdout.write(status)
    sys.stdout.flush()


# Load each image file
def loadImage(filename):
    img = Image.open(filename)
    return np.asarray(img.getdata())


# Load training data from file
def loadTrainingData():
    print ('Loading training data...')
    X = []
    Y = []
    test_rate = 0.1
    classes = 10
    class_data = 1000

    for i in range(0, classes):
        for j in range(0, class_data):
            label = [0] * classes
            label[i] = 1
            label = [float(k) for k in label]
            image = loadImage(train_path + str(i) + '/' + str(i) + '_' + str(j) + '.png')
            image = [float(k) for k in image]
            X.append(image)
            Y.append(label)
            printStatus("\r" + str(i) + ': (' + str(j) + '/1000)')
    print
    X, validX, Y, validY = train_test_split(X, Y, test_size=test_rate)
    return (np.array(X), np.array(Y), np.array(validX), np.array(validY))


# Load testing data from file
def loadTestingData():
    print ('Loading testing data...')
    testX = []
    for i in range(0, 10000):
        image = loadImage(test_path + str(i) + '.png')
        image = [float(k) for k in image]
        testX.append(image)
        printStatus("\r" + str(i) + '/10000')
    print
    return np.array(testX)


# Load training data and testing data
print ("Ready to start.")
X, Y, validX, validY = loadTrainingData()
testX = loadTestingData()


# Transform input data
X = X.reshape([-1, 28, 28, 1])
validX = validX.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])


# Construct deep learning network
print ("Ready to build layers.")
network = input_data(shape=[None, 28, 28, 1], name='input')

network = conv_2d(network, 32, 3, activation='relu')
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = local_response_normalization(network)

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = local_response_normalization(network)

network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2)
network = local_response_normalization(network)

network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='target')


# Train model with training data
model = tflearn.DNN(network, tensorboard_verbose=0)
if os.path.isfile(model_file):
    print ("Get model from file.")
    model.load(model_file)
else:
    print ("Ready to get fit.")
    model.fit({'input': X}, {'target': Y},
              n_epoch=50,
              batch_size=100,
              validation_set=({'input': validX}, {'target': validY}),
              snapshot_step=100,
              show_metric=True,
              run_id='convnet_mnist')
    model.save(model_file)


# Predict testing data
predictedY = np.array(model.predict(testX))
predictedY = np.argmax(predictedY, axis=1)


# Print out predicted result
summary = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
fstream = open(result_file, 'w+')
for i in range(0, len(predictedY)):
    fstream.write(str(predictedY[i]) + '\n')
    summary[predictedY[i]] += 1
fstream.close()
print summary
