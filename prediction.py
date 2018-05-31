from sklearn import svm, metrics
from PIL import Image
import numpy as np
import sys

TEST_DIR = './hw4_test/'
TRAIN_DIR = './hw4_train/'
RESULT_FILE = 'prediction.txt'

def main():
    train_set, target_set = loadTrainingData()
    test_set = loadTestingData()

    print ('Training data...')
    n = len(train_set)
    classifier = svm.SVC(gamma=0.001)
    classifier.fit(train_set, target_set)

    print ('Predicting data...')
    predicted = classifier.predict(test_set)

    print ('Outputing prediction...')
    fstream = open(RESULT_FILE, 'w+')
    for i in range(0, n):
        fstream.write(str(predicted[i]) + '\n')
    fstream.close()

    print ('Done!')


def loadTrainingData():
    print ('Loading training data...')
    train_set = []
    target_set = []
    for i in range(0, 10):
        for j in range(0, 1000):
            train_set.append(loadImage(TRAIN_DIR + str(i) + '/' + str(i) + '_' + str(j) + '.png'))
            target_set.append(i)
            printStatus("\r" + str(i) + ': (' + str(j) + '/1000)')
    print
    return train_set, target_set

def loadTestingData():
    print ('Loading testing data...')
    test_set = []
    for i in range(0, 10000):
        test_set.append(loadImage(TEST_DIR + str(i) + '.png'))
        printStatus("\r" + str(i) + '/10000')
    print
    return test_set

def printStatus(status):
    sys.stdout.write(status)
    sys.stdout.flush()

def loadImage(filename):
    img = Image.open(filename)
    return np.asarray(img.getdata())

if __name__ == "__main__":
    main()
















# https://www.wolfib.com/Image-Recognition-Intro-Part-1/
# http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html


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

