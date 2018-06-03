# https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_mnist.py

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import sys


test_path = './hw4_test/'
train_path = './hw4_train/'
result_file = 'prediction.txt'


def printStatus(status):
    sys.stdout.write(status)
    sys.stdout.flush()

def loadImage(filename):
    img = Image.open(filename)
    return np.asarray(img.getdata())

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
    X, validX, Y, validY = train_test_split(X, Y, test_size=0.1, random_state=42)
    return (np.array(X), np.array(Y), np.array(validX), np.array(validY))

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


# Data loading and preprocessing
print ("Ready to start.")
X, Y, validX, validY = loadTrainingData()
testX = loadTestingData()

X = X.reshape([-1, 28, 28, 1])
validX = validX.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])


# Building convolutional network
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

# Training
print ("Ready to get fit.")
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y},
          n_epoch=200,
          shuffle=True,
          batch_size=50,
          validation_set=({'input': validX}, {'target': validY}),
          snapshot_step=100,
          show_metric=True,
          run_id='convnet_mnist')


predictedY = np.array(model.predict(testX))
predictedY = np.argmax(predictedY, axis=1)

summary = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
fstream = open(result_file, 'w+')
for i in range(0, len(predictedY)):
    fstream.write(str(predictedY[i]) + '\n')
    summary[predictedY[i]] += 1
fstream.close()

print summary

# correctRatio = np.mean(np.equal(np.argmax(predictedY, axis=1), np.argmax(Y, axis=1)).astype(int))
# print("Validation accuracy using predict (max): {}".format(correctRatio))
