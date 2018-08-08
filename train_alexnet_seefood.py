"""
With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset.
Specify the configuration settings at the beginning according to your
problem.

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Food data from:
Not-Food data from:

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com

Changed by: Ning Xie
contact: xie.25@wright.edu

Adapted for CEG 4110 by: Derek Doran
contact: derek.doran@wright.edu
Aug 3 2017
"""

from __future__ import print_function
import random
import os
import numpy as np
from datetime import datetime
import tensorflow as tf
from PIL import Image

from alexnet import AlexNet
#import tf_cnnvis
import process_food_data

# Learning and network params
learning_rate = 0.001
num_epochs = 20
batch_size = 750
dropout_rate = 0.5
num_classes = 2  # data output dimension
train_layers = ['fc8','fc7', 'fc6']
# How often we want to write the tf.summary data to disk
display_step = 1
cnn_vis = False
# Early Stopping parameters
early_stop_flag = 1

# run "tensorboard --logdir=tensorflow_save" in terminal after training

#sess = tf.InteractiveSession()

# Create directory for auto saving
dir_path = "tensorflow_save/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# xrange is not available in Python versions 3+
try:  # try to...
    xrange  # find a home, home on the xrange
except NameError:  # if it's not available
    xrange = range  # create a variable for it


###############################################################################################
###################################### Prepare the Dataset ####################################
###############################################################################################

# Training + Testing + Validation Data
rootdir = os.getcwd()+"/data"


# # DATA AUGMENTATION
# # NING'S NOTE: ONLY UNCOMMENT THIS FOR DATA AUGMENTATION.
# # PLEASE BE SURE TO COMMENT THIS IF THE DATA IS ALREADY AUGMENTATED!
# from data_aug import dataAugmentation
# dataAugmentation(rootdir)

image_width = 227
image_height = 227

all_data_collection = process_food_data.getData(rootdir, percent_food_to_use=0.66)

if len(all_data_collection[0]) == 0:
    print("Error : \'rootdir\' path is incorrect! Please check!")
    exit()

# Get the paths of all images
image_paths = all_data_collection[0]
# Get the scene label of all images
image_labels = all_data_collection[1]
# this is a list of unique scenes represented in image_scene
unique_labels= list(set(image_labels))

target_size = 2
data_size = len(image_paths)
# train:val:test = 6:2:2
each_size = int(data_size * 1.0 / 10)
train_size = each_size * 6
val_size = each_size * 2
test_size = data_size - train_size - val_size

image_path_tensor = tf.convert_to_tensor(image_paths, dtype=tf.string)


#####################
#### Model Config ####
#####################

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "tensorflow_save"
checkpoint_path = "tensorflow_save/ckpt"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

if early_stop_flag:
    # Best validation accuracy seen so far.
    best_val_acc = 0.0
    # Iteration-number for last improvement to validation accuracy.
    last_improve = 0
    # Stop optimization if no improvement found in this many iterations.
    patience_improve = 3
    total_iterations = 0

#####################
#### Model Construction ####
#####################

# tf Graph input
with tf.name_scope('Input_xn'):
    x = tf.placeholder(tf.float32, [None, image_width, image_height, 3])
with tf.name_scope('Output_label_xn'):
    y = tf.placeholder(tf.float32, [None, num_classes])

# Dropout
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8
# tf.summary.histogram('output', score)
pred = tf.argmax(score, 1)
tf.summary.histogram('output_class', pred)

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("Cross_entropy_xn"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))
    # loss = tf.reduce_mean(tf.losses.hinge_loss(logits = score, labels = y))
    # REGULARIZATION
    with tf.variable_scope('fc6', reuse=True) as scope:
        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights')
        loss = loss + 2.0 * tf.nn.l2_loss(weights)
    with tf.variable_scope('fc7', reuse=True) as scope:
        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights')
        loss = loss + 2.0 * tf.nn.l2_loss(weights)

# Train op
with tf.name_scope("Optimizer_xn"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: accuracy_top_1 of the model
with tf.name_scope("accuracy_top_1"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy_top_1 = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy_top_1 to the summary
tf.summary.scalar('accuracy_top_1', accuracy_top_1)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Start Tensorflow session
# By growth (like theano)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# A Reader to process the image files
reader = tf.WholeFileReader()

#####################
#### Running Model ####
#####################
print("[{}] Spooling up tensorflow session for training...".format(datetime.now()))
with tf.Session(config=config) as sess:

    partitions = [0] * data_size
    partitions[:val_size] = [1] * val_size
    partitions[val_size:val_size + test_size] = [2] * test_size
    random.shuffle(partitions)
    # partition our data into a train, test, validation set according to our partition vector
    train_image_paths, val_image_paths, test_image_paths = tf.dynamic_partition(image_path_tensor, partitions, 3)

    # Initialize the FileWriters
    train_writer = tf.summary.FileWriter(filewriter_path + '/train', sess.graph)
    val_writer = tf.summary.FileWriter(filewriter_path + '/val')
    test_writer = tf.summary.FileWriter(filewriter_path + '/test')

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # # Restore variables from disk. There should not be any init!!!!
    # saver.restore(sess, "tensorflow_save/model.ckpt")
    # print "Model restored."
    # # Do some work with the model

    # Add the model graph to TensorBoard
    # writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    # Config and coordinators for image file reading
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    coord = tf.train.Coordinator()

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    ##################################
    # Loop over number of epochs
    for epoch in range(num_epochs):

        total_iterations += 1

        print("************************** New Epoch ***************************")
        print("{} Epoch number: {}".format(datetime.now(), epoch))

        total_batch = train_size / batch_size + 1 #train_size

        # shuffle the order of the training, validation, test data
        train_x_shuffled = tf.random_shuffle(train_image_paths)
        val_x_shuffled = tf.random_shuffle(val_image_paths)
        test_x_shuffled = tf.random_shuffle(test_image_paths)

        for i in range(total_batch):
            if i < total_batch - 1:
                batch_files = train_x_shuffled[i * batch_size: (i + 1) * batch_size].eval()
            else:
                if train_size % batch_size:
                    batch_files = train_x_shuffled[i * batch_size: ].eval()
                else:
                    break

            ### Build the mini-batch of input tensors
            print("[{}] Building mini batch...".format(datetime.now()))
            with tf.device('/cpu:0'):
                batch_x = []
                batch_y = []
                for path in batch_files:
                    image = Image.open(path)
                    image = image.resize((image_width, image_height), Image.BILINEAR)
                    tensor = np.asarray(image, dtype=np.float32)
                    if len(tensor.shape) == 3:
                        batch_x.append(np.asarray(image, dtype=np.float32))
                        ## [1 0] means food; [0 1] means NOT food
                        not_food = int('not-food' in path)
                        batch_y.append([int(not not_food), not_food])

            print("[{}] Running backprop...".format(datetime.now()))
            sess.run(train_op, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file
            if i % display_step == 0:
                merged_s, loss_return = sess.run([merged_summary, loss],
                                                 feed_dict={x: batch_x,
                                                            y: batch_y,
                                                            keep_prob: 1.})
                train_writer.add_summary(merged_s, epoch * batch_size + i)

                print("Epoch:", '%04d' % epoch,
                      "batch:", '%04d' % i,
                      "batch loss:", "{:.6f}".format(loss_return))

        # #########################################################################################
        # ##################### Evaluation on Validation & Testing Dataset ########################
        # #########################################################################################

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        val_acc_top_1 = 0.
        val_loss = 0.
        val_count = 0

        total_batch = val_size / batch_size + 1

        for i in range(total_batch):
            if i < total_batch - 1:
                batch_files = val_x_shuffled[i * batch_size: (i + 1) * batch_size].eval()
            else:
                if train_size % batch_size:
                    batch_files = val_x_shuffled[i * batch_size: len(val_x_shuffled)].eval()
                else:
                    break

            ### Build the mini-batch of input tensors
            print("[{}] Building mini batch...".format(datetime.now()))
            with tf.device('/cpu:0'):
                batch_x = []
                batch_y = []
                for path in batch_files:
                    image = Image.open(path)
                    image = image.resize((image_width, image_height), Image.BILINEAR)
                    tensor = np.asarray(image, dtype=np.float32)
                    if len(tensor.shape) == 3:
                        batch_x.append(np.asarray(image, dtype=np.float32))
                        ## [1 0] means food; [0 1] means NOT food
                        not_food = int('not-food' in path)
                        batch_y.append([int(not not_food), not_food])

            acc_top_1, los, pre = sess.run([accuracy_top_1, loss, pred],
                                                      feed_dict={x: batch_x,
                                                                 y: batch_y,
                                                                 keep_prob: 1.})
            print("Validation Prediction:")
            print(pre)
            val_acc_top_1 += acc_top_1
            val_loss += los
            val_count += 1

            if i % display_step == 0:
                # xn-CNN visualization
                merged_s, loss_return = sess.run([merged_summary, loss],
                                                 feed_dict={x: batch_x,
                                                            y: batch_y,
                                                            keep_prob: 1.})
                val_writer.add_summary(merged_s, epoch * batch_size + i)

                print("Validation Epoch:", '%04d' % epoch,
                      "Validation batch:", '%04d' % i,
                      "Validation batch loss:", "{:.6f}".format(loss_return))


        val_acc_top_1 /= val_count
        val_loss /= val_count
        print("Validation loss:", "{:.6f}".format(val_loss),
              "Validation accuracy_top_1:", "{:.6f}".format(val_acc_top_1))

        # Early Stopping
        if early_stop_flag:
            if val_acc_top_1 > best_val_acc:
                # Update the best-known validation accuracy.
                best_val_acc = val_acc_top_1
                # Set the iteration for the last improvement to current.
                last_improvement = total_iterations

            # If no improvement found in the required number of iterations.
            if total_iterations - last_improvement > patience_improve:
                print("No improvement found in a while, stopping optimization.")
                # Break out from the for-loop.
                break

        print("{} Saving checkpoint of model...".format(datetime.now()))
        # save checkpoint of the model
        # if epoch % 5 == 0:
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)
        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

    print("************************** Training Finished! **************************")

    print("{} Start testing".format(datetime.now()))

    # Test model
    test_acc_top_1 = 0.
    test_loss = 0.
    test_count = 0

    total_batch = test_size / batch_size + 1

    for i in range(total_batch):
        if i < total_batch - 1:
            batch_files = test_x_shuffled[i * batch_size: (i + 1) * batch_size].eval()
        else:
            if train_size % batch_size:
                batch_files = test_x_shuffled[i * batch_size: len(test_x_shuffled)].eval()
            else:
                break

        ### Build the mini-batch of input tensors
        print("[{}] Building mini batch...".format(datetime.now()))
        with tf.device('/cpu:0'):
            batch_x = []
            batch_y = []
            for path in batch_files:
                image = Image.open(path)
                image = image.resize((image_width, image_height), Image.BILINEAR)
                tensor = np.asarray(image, dtype=np.float32)
                if len(tensor.shape) == 3:
                    batch_x.append(np.asarray(image, dtype=np.float32))
                    ## [1 0] means food; [0 1] means NOT food
                    not_food = int('not-food' in path)
                    batch_y.append([int(not not_food), not_food])

        acc_top_1, acc_top_3, los, pre = sess.run([accuracy_top_1, loss, pred],
                                                  feed_dict={x: batch_x,
                                                             y: batch_y,
                                                             keep_prob: 1.})
        print("Test Prediction:")
        print(pre)
        test_acc_top_1 += acc_top_1
        test_loss += los
        test_count += 1

        if i % display_step == 0:
            merged_s, loss_return = sess.run([merged_summary, loss],
                                             feed_dict={x: batch_x,
                                                        y: batch_y,
                                                        keep_prob: 1.})
            test_writer.add_summary(merged_s, epoch * batch_size + i)

            print("Testing Epoch:", '%04d' % epoch,
                  "Testing batch:", '%04d' % i,
                  "Testing batch loss:", "{:.6f}".format(loss_return))

    test_acc_top_1 /= test_count
    test_loss /= test_count

    print("Testing loss:", "{:.6f}".format(test_loss),
          "Testing accuracy_top_1:", "{:.6f}".format(test_acc_top_1))