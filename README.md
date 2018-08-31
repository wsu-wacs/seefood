# CEG4110 Fall 2018 Project: SeeFood
You see food, I see food... can your phone see food? It better if you want to get an A in CEG 4110 this semester! 

For this semester's project you will implement a software product (written in the language of your choosing) that is able to ingest an image and tell the user whether or not it has food. The project involves: 
- hosting a deep learning model in the cloud (Amazon Web Services) on an EC2 instance; 
- writing an API interacts with <b>SeeFood-Core-AI</b>, a deep learning model written in TensorFlow that has been trained to identify images of food; 
- hosting the API on an AWS instance; 
- writing a software program (ideally for iOS or Android) that connects to your hosted API and uses SeeFood-Core-AI to tell a user if a provided image is a picture of food; 
- satisfying a number of other functional requirements for the software program.

## What is SeeFood-Core-AI?
SeeFood-Core-AI [SFCA] is a machine learning model that has been trained to recognize if an image prominantley features
food (of any type). SFCA taught itself to learn the features, patterns, colors, and textures of images that have and
does not have food at a wide range of levels of generality. To learn, SFCA was exposed to images of food from the 
Food-101 dataset, which has about 100k images of 100 different types of food. Of course, SFCA also needs to learn 
some properties of images that are not food, and for this purpose SFCA was exposed to the Caltech-256 dataset of ~36k images across 256 different objects. Food was randomly subsampled so that SFCA saw about as much food as non-food. The images can be browsed at the links below. 

SFCA is a typle of Convolutional Neural Network (CNN). Students that have taken a course in machine learning or soft computing, or is familiar with neural networks, can find more details about CNNs below. In essence, the 
model learns to identify particular patterns that exist in small patches of images that are (not) of food. The model then learns 'patterns of patterns', and then 'patterns of patterns of patterns', essentially learning ever more abstract qualities of images featuring food. It is a deep learning model, whose architecture follows AlexNet, and was trained over a 24 hour period on an nvidia Titan X graphics card. The model was implemented in TensorFlow. The source code for training and testing the model is available in this repository. 

On a test data set composed of a mixture of ~8k images from Food-101 and Caltech-256, SFCA achieves 96% accuracy in finding food in an image. 

<b> You do not need a background in machine learning or an understanding of neural networks to use SFCA!</b> We provide code that loads a trained SFCA model, ingests the path to an image from a python command line, and returns a message indicating if food (or no food) can be found in the image. The script is provided <b>as is</b> with <b>no exception handling or checks on the input</b>. It naively assumes that the path provided exists and points to an image file. The script will crash if the path does not exist, the image is corrupt, or the file is not an image. 

### References
Food-101 Dataset: https://www.vision.ee.ethz.ch/datasets_extra/food-101/

Caltech-256 Dataset: http://www.vision.caltech.edu/Image_Datasets/Caltech256/

Nvidia's Gentle Intro to Deep Learning: https://devblogs.nvidia.com/parallelforall/deep-learning-nutshell-core-concepts/

Convolutional Neural Network Overview: http://cs231n.github.io/convolutional-networks/ 

AlexNet: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

## Summary of Requirements
SFCA requires Python 2.7 (<b>not</b> compatible with Python3). The python dependencies for running the trained SFCA to recognize images (find_food.py) are
- numpy
- tensorflow
- pillow

These packages are available via the pip package manager and are already installed on the Amazon EC2 image we provide. 

## Data Repositories
There are other food image datasets in the wild! Check out: 

Food-5k and Food-11: http://mmspg.epfl.ch/food-image-datasets

CMU and Intel's Fast Food dataset: http://pfid.rit.albany.edu/index.php


## Supporting Files
Some required files are larger than the limit of what can be uploaded to GitHub. Follow the Google Drive link
below to download the following files: 
- bvlc_alexnet.npy : this is a set of model parameters needed to train SFCA. You do <b> not </b> need this file unless you would like to (independently) experiment with running SFCA. 
- saved_model.zip : this is a zip of the trained SFCA model. You will need to download and deploy this on Amazon EC2 to perform food recognition. 

<b> Link to Supporting Files: </b> https://drive.google.com/drive/folders/0B7rFjZJTwj0oVkZzcWJvbnJybU0?usp=sharing

## Amazon EC2
Because SFCA has tremendous memory (about 750MB of RAM) and processing requirements, it is not feasible to run it on a client machine, and especially not in a web browser or smartphone. Perhaps the only client-side application that can get away with consuming 750MB+ of RAM is a computer game or scientific computing software where the user anticipates high memory usage.

To address this problem, you will deply SFCA to the cloud - specifically to an Amazon Web Service EC2 compute node. An image of this compute node has already been precompiled and is available for you via... The image is running Amazon Linux (AMI); see 
https://aws.amazon.com/amazon-linux-ami/ for more details. 

You will deploy SFCA as a <b> web application </b>, hosted on your EC2 compute node. You will also need to implement an API that allows a thin client to make http requests to the compute node with an image for SFCA to process. The API should return SFCA's class label. This will let you implement your software on even a resource starved device, e.g. on a web browser (via Javascript) or an Andriod/iOS phone (via Java) without consume the resources of a user's device. The script for demonstrating how to run the trained SFCA model (`find_food.py`) provided in this repo will need to undergo significant modification to achieve these goals. 

By the end of the semester you'll certianly know more about linux, python, and web services than you know now! You do not need to be a guru or be completely comfortable writing python code or working in linux, and in fact you can even be successful if you know nothing about them. But YES, you will be working in an ssh terminal, connected to a linux machine (the EC2 instance), loading and running python scripts and doing some system configurations. YES, you are going to run a python-based web app SFCA will run within, and you are going to need to write Python, to modify see-food.py to make it robust to user input and to massage its output to fit your application, and you are going to need to play around and try and think on your feet and use stackoverflow and all the rest of it to figure out what to do. 

There are no detailed instructions or tutorials being offered to help you here. The good news is we are all upper division CS/CEG students, carrying a certain degree of computing experience, intuition, and background that makes self-learning and troubleshooting of the issues you'll face possible. And of course StackOverflow is a thing!

<a href="url"><img src="https://images.gr-assets.com/books/1457364208l/29437996.jpg" align="center" height="400" width="300" ></a>
 
It is completely possible for you to learn what is necessary on your own to complete this project. In fact this is a big opportunity for you. If <b> YOU </b> put in the time to learn EC2, to learn some Python, to learn about hosting Web apps for this project, you'll obtain fantastic keywords to add to your resume, and you will be able to talk about your exposure to what are <b> very relevant </b> technologies for practicing software engineers. Do <b> not </b> be the person who gives up out for fear of needing to learn or lose ignorance, do <b> not </b> be the person in the background being carried. Be the carry. 

### Hosting SFCA As a Web Application
You will ultimately need to run SFCA within a Python web app that lives on the AWS EC2 instance. That application will have an API your client will call (via HTTP[s]). The API needs, at a minimum, functionality to
- upload an image to the application
- retrieve an image classification (food or not food) for the image sent

<b> You are responsible for deciding how to architect the web application. </b> But you definitely want to leverage an existing Python web framework. A number of choices are available here: https://wiki.python.org/moin/WebFrameworks

Derek strongly recommends that you look at Flask, in particular, as it is very simple to use: http://flask.pocoo.org

See for example a simple Flask web service here: http://flask.pocoo.org/docs/0.12/quickstart/

## Running SeeFood
One of the first things you should do is give SFCA a whirl locally. Making sure Python 2.7 and the above mentioned dependencies are satisfied (numpy, tensorflow, pillow), download `SFCA_trained.zip` and `find_food.py`. Extract the contents of `SFCA_trained.zip` to a subdirectory `saved_model`. `find_food.py` must be located in the parent directory of `saved_model`. 

Lets try some examples, like the loaded fries image in the samples directory of the repository: 
<a href="url"><img src="https://s-media-cache-ak0.pinimg.com/originals/08/51/f0/0851f082e4177a2e706ba3f53074c975.jpg" align="center" height="400" width="350" ></a>

From a terminal in the directory `find_food.py' is in: 

```
python find_food.py samples/loaded_fries.png
```

If you see some tensorflow errors about compiling to use CPU instructions don't worry about it. After some time we get the output: 
```
[[2.62415957 -0.55939353]]
Oh yes... I see food! :D
```
The numbers in the double hard brackets are scores assigned to the choice that the image is of food (the left number) versus not of food (the right number). These numbers are not interpretable, except that the higher number corresponds to SFCA's decision, and the difference of the two numbers conveys how confident we can be about the decision. 

How about Food Network star Giada De Laurentiis making a very nice salad, any food here? 

<a href="url"><img src="http://food.fnr.sndimg.com/content/dam/images/food/editorial/talent/giada-de-laurentiis/FN_Giada-De-Laurentiis-About.jpg.rend.hgtvcom.336.336.suffix/1457731981255.jpeg" align="center"></a>

```
python find_food.py samples/giada.png
```
The result: 
```
[[-0.23845851 2.30115366]]
No food here... :(
```
Despite the salad, it is hard to argue that this is really an image of food; this is more like a portrait of a person. And indeed, taking the absolute value of the difference of the scores we see that SFCA is a little less confident about its choice compared to the loaded fries.

Try out some other samples. If you look at the high_school_lunch_room image in the samples directory: 

<a href="url"><img src="http://interculturaltalk.org/wp-content/uploads/2015/09/High-School-Lunch-Room.jpg" align="center"></a>

you'll find that SFCA sees food with scores `[[1.37732399 0.68766689]]`. A relatively close decision on a tricky image -- what do you think is it an image of? The food the students are eating, or the students eating the food? SFCA says the food is the better topic.

You will note that it takes a long time to recognize an image as the code needs to load the pre-trained model into memory (about 750MB!). You will need to figure out how to have he model persist in memory on AWS, such that queries from your application will be much faster. 

