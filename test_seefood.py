"""
A script to test a trained SeeFood model using lots of images. 
Script assumes that the data for testing is organized the same way
as for training. 

For CEG 4110 by: Derek Doran
contact: derek.doran@wright.edu
Aug 9 2017
"""
import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime
import process_food_data

###### Initialization code - we only need to run this once and keep in memory.
sess = tf.Session()
saver = tf.train.import_meta_graph('saved_model/model_epoch5.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('saved_model/'))
graph = tf.get_default_graph()
x_input = graph.get_tensor_by_name('Input_xn/Placeholder:0')
keep_prob = graph.get_tensor_by_name('Placeholder:0')
class_scores = graph.get_tensor_by_name("fc8/fc8:0")
######

# Training + Testing + Validation Data
rootdir = os.getcwd()+"/data"

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

batch_size = 100

partitions = [0] * data_size
partitions[:val_size] = [1] * val_size
partitions[val_size:val_size + test_size] = [2] * test_size
random.shuffle(partitions)
# partition our data into a train, test, validation set according to our partition vector
train_image_paths, val_image_paths, test_image_paths = tf.dynamic_partition(image_path_tensor, partitions, 3)

test_x_shuffled = tf.random_shuffle(test_image_paths)

print("{} Start testing".format(datetime.now()))

# Test model
test_acc_top_1 = 0.
test_count = 0

total_batch = test_size / batch_size + 1

for i in range(total_batch):
    if i < total_batch - 1:
        batch_files = sess.run(test_x_shuffled[i * batch_size: (i + 1) * batch_size])
    else:
        if train_size % batch_size:
            batch_files = sess.run(test_x_shuffled[i * batch_size: len(test_x_shuffled)])
        else:
            break

    ### Build the mini-batch of input tensors
    for path in batch_files:
        image = Image.open(path).convert('RGB')
        image = image.resize((227, 227), Image.BILINEAR)
        tensor = np.asarray(image, dtype=np.float32)
        if len(tensor.shape) == 3:
            x_in = [tensor]
            ## [1 0] means food; [0 1] means NOT food
            not_food = int('not-food' in path)
            y = [int(not not_food), not_food]
            score = sess.run(class_scores, {x_input: x_in, keep_prob: 1.})
            if np.argmax(score) == np.argmax(y):
                test_acc_top_1 += 1
            test_count +=1
    print("Current Accuracy: {:.4f}".format(test_acc_top_1 / test_count))
    print("# Tests so far: {}".format(test_count))

print("Final Accuracy: {:.4f}".format(test_acc_top_1 / test_count))