"""
A script to ask SeeFood if it sees food in the image at 
path specified by the command line argument.

For CEG 4110 by: Derek Doran
contact: derek.doran@wright.edu
Aug 9 2017
"""
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

parser = argparse.ArgumentParser(description="Ask SeeFood if there is food in the image provided.")
parser.add_argument('image_path', help="The full path to an image file stored on disk.")
args = parser.parse_args()

# The script assumes the args are perfect, this will crash and burn otherwise.

###### Initialization code - we only need to run this once and keep in memory.
sess = tf.Session()
saver = tf.train.import_meta_graph('saved_model/model_epoch5.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('saved_model/'))
graph = tf.get_default_graph()
x_input = graph.get_tensor_by_name('Input_xn/Placeholder:0')
keep_prob = graph.get_tensor_by_name('Placeholder:0')
class_scores = graph.get_tensor_by_name("fc8/fc8:0")
######

# Work in RGBA space (A=alpha) since png's come in as RGBA, jpeg come in as RGB
# so convert everything to RGBA and then to RGB.
image_path = args.image_path
image = Image.open(image_path).convert('RGB')
image = image.resize((227, 227), Image.BILINEAR)
img_tensor = [np.asarray(image, dtype=np.float32)]
print 'looking for food in '+image_path

#Run the image in the model.
scores = sess.run(class_scores, {x_input: img_tensor, keep_prob: 1.})
print scores
# if np.argmax = 0; then the first class_score was higher, e.g., the model sees food.
# if np.argmax = 1; then the second class_score was higher, e.g., the model does not see food.
if np.argmax(scores) == 1:
    print "No food here... :( "
else:
    print "Oh yes... I see food! :D"
