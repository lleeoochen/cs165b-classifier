from sklearn import svm, metrics
from PIL import Image
import numpy as np
import sys
import tensorflow as tf

test_path = './hw4_test/'
train_path = './hw4_train/'
result_file = 'prediction.txt'

img_size = 28
num_channels = 3
filter_size_conv1 = num_channels
filter_size_conv2 = num_channels
filter_size_conv3 = num_channels
num_filters_conv1 = 28
num_filters_conv2 = 28
num_filters_conv3 = 56
fc_layer_size = 112

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_classes = len(classes)
validation_size = 0.2
batch_size = 20
total_iterations = 0


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations, total_iterations + num_iteration):
 
        if (i + 1) >= total_iterations + num_iteration:
            x_batch = data['train']['contents'][i * batch_size:]
            y_true_batch = data['train']['labels'][i * batch_size:]
        else:
            x_batch = data['train']['contents'][i * batch_size: (i + 1) * batch_size]
            y_true_batch = data['train']['labels'][i * batch_size: (i + 1) * batch_size]
        
        if (i + 1) >= total_iterations + num_iteration:
            x_valid_batch = data['valid']['contents'][i * batch_size:]
            y_valid_batch = data['valid']['labels'][i * batch_size:]
        else:
            x_valid_batch = data['valid']['contents'][i * batch_size: (i + 1) * batch_size]
            y_valid_batch = data['valid']['labels'][i * batch_size: (i + 1) * batch_size]

        print len(x_batch)
        feed_dict_tr = {x: x_batch, y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}
        session.run(optimizer, feed_dict=feed_dict_tr)
 
        if i % int(len(data['train']['contents'])/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / len(data['train']['contents']) / batch_size)    
            
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, 'my-model') 
 
 
    total_iterations += num_iteration

def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):  
    
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])

    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    ## We shall be using max-pooling.  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])

    return layer

def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def loadTrainingData():
    print ('Loading training data...')
    X = []
    Y = []
    testX = []
    testY = []
    test_rate = 0.2
    classes = 10
    class_data = 300

    for i in range(0, classes):
        for j in range(0, class_data):
            label = [0] * classes
            label[i] = 1
            label = [float(k) for k in label]
            image = loadImage(train_path + str(i) + '/' + str(i) + '_' + str(j) + '.png')
            image = [float(k) for k in image]
            if j * 1.0 / class_data < test_rate:
                testX.append(image)
                testY.append(label)
            else:
                X.append(image)
                Y.append(label)

            printStatus("\r" + str(i) + ': (' + str(j) + '/1000)')
    print

    return {
        'train': {
            'contents': np.array(X),
            'labels': np.array(Y)
        },
        'valid': {
            'contents': np.array(testX),
            'labels': np.array(testY)
        }
    }

def loadTestingData():
    print ('Loading testing data...')
    test_set = []
    for i in range(0, 10000):
        test_set.append(loadImage(test_path + str(i) + '.png'))
        printStatus("\r" + str(i) + '/10000')
    print
    return test_set

def printStatus(status):
    sys.stdout.write(status)
    sys.stdout.flush()

def loadImage(filename):
    img = Image.open(filename)
    return np.asarray(img.getdata())


data = loadTrainingData()
print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data['train']['labels'])))
print("Number of files in Validation-set:\t{}".format(len(data['valid']['labels'])))

a = tf.truncated_normal([100, 28, 28, 3])
session = tf.Session()
session.run(tf.global_variables_initializer())
session.run(tf.shape(a))

b = tf.reshape(a,[100, 28*28*3])
session.run(tf.shape(b))

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

print('layer1')
layer_conv1 = create_convolutional_layer(
    input=x,
    num_input_channels=num_channels,
    conv_filter_size=filter_size_conv1,
    num_filters=num_filters_conv1)

print('layer2')
layer_conv2 = create_convolutional_layer(
    input=layer_conv1,
    num_input_channels=num_filters_conv1,
    conv_filter_size=filter_size_conv2,
    num_filters=num_filters_conv2)

print('layer3')
layer_conv3= create_convolutional_layer(
    input=layer_conv2,
    num_input_channels=num_filters_conv2,
    conv_filter_size=filter_size_conv3,
    num_filters=num_filters_conv3)

print('layer flat')
layer_flat = create_flatten_layer(layer_conv3)

print('layer fc1')
layer_fc1 = create_fc_layer(
    input=layer_flat,
    num_inputs=layer_flat.get_shape()[1:4].num_elements(),
    num_outputs=fc_layer_size,
    use_relu=True)

print('layer fc2')
layer_fc2 = create_fc_layer(
    input=layer_fc1,
    num_inputs=fc_layer_size,
    num_outputs=num_classes,
    use_relu=False)

y_pred = tf.nn.softmax(layer_fc2, name="y_pred")
y_pred_cls = tf.argmax(y_pred, axis=1)
session.run(tf.global_variables_initializer())

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
print (cross_entropy.shape)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session.run(tf.global_variables_initializer()) 

saver = tf.train.Saver()
train(num_iteration=3000)


# http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/