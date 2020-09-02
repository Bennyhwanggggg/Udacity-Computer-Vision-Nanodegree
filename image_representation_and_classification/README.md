# Image Representation and Classification
## Computer Vision Pipeline
A computer vision pipeline is a series of steps that most computer vision applications will go through. Many vision applications start off by acquiring images and data, then processing that data, performing some analysis and recognition steps, then finally performing an action. The general pipeline is pictured below!

**General computer vision processing pipeline**
![Image](https://video.udacity-data.com/topher/2017/March/58d03bda_screen-shot-2017-03-13-at-12.36.54-pm/screen-shot-2017-03-13-at-12.36.54-pm.png)

Now, let's take a look at a specific example of a pipeline applied to facial expression recognition.

**Facial recognition pipeline**
![Image](https://video.udacity-data.com/topher/2018/April/5ade67ef_screen-shot-2018-04-23-at-4.10.20-pm/screen-shot-2018-04-23-at-4.10.20-pm.png)

### Standardizing Data
Pre-processing images is all about standardizing input images so that you can move further along the pipeline and analyze images in the same way. In machine learning tasks, the pre-processing step is often one of the most important.

For example, imagine that you've created a simple algorithm to distinguish between stop signs and other traffic lights.

![Image](https://video.udacity-data.com/topher/2018/April/5ade672f_screen-shot-2018-04-23-at-4.05.20-pm/screen-shot-2018-04-23-at-4.05.20-pm.png)

Images of traffic signs; a stop sign is on top and a hiking sign is on the bottom.

If the images are different sizes, or even cropped differently, then this counting tactic will likely fail! So, it's important to pre-process these images so that they are standardized before they move along the pipeline. In the example below, you can see that the images are pre-processed into a standard square size.

The algorithm counts up the number of red pixels in a given image and if there are enough of them, it classifies an image as a stop sign. In this example, we are just extracting a color feature and skipping over selecting an area of interest (we are looking at the whole image). In practice, you'll often see a classification pipeline that looks like this.

![Image](https://video.udacity-data.com/topher/2018/April/5ade6541_stop-sign-classification/stop-sign-classification.png)

## Training a Neural Network
To train a computer vision neural network, we typically provide sets of labelled images, which we can compare to the predicted output label or recognition measurements. The neural network then monitors any errors it makes (by comparing the correct label to the output label) and corrects for them by modifying how it finds and prioritizes patterns and differences among the image data. Eventually, given enough labelled data, the model should be able to characterize any new, unlabeled, image data it sees!

A training flow is pictured below. This is a convolutional neural network that learns to recognize and distinguish between images of a smile and a smirk.
![Image](https://video.udacity-data.com/topher/2018/April/5ade68dd_screen-shot-2018-04-23-at-4.14.19-pm/screen-shot-2018-04-23-at-4.14.19-pm.png)

**Gradient descent** is a a mathematical way to minimize error in a neural network. More information on this minimization method can be found here.

**Convolutional neural networks** are a specific type of neural network that are commonly used in computer vision applications. They learn to recognize patterns among a given set of images.

### Machine Learning and Neural Networks
When we talk about machine learning and neural networks used in image classification and pattern recognition, we are really talking about a set of algorithms that can learn to recognize patterns in data and sort that data into groups.

The example we gave earlier was sorting images of facial expressions into two categories: smile or smirk. A neural network might be able to learn to separate these expressions based on their different traits; a neural network can effectively learn how to draw a line that separates two kinds of data based on their unique shapes (the different shapes of the eyes and mouth, in the case of a smile and smirk). Deep neural networks are similar, only they can draw multiple and more complex separation lines in the sand. Deep neural networks layer separation layers on top of one another to separate complex data into groups.

### Separating Data
Say you want to separate two types of image data: images of bikes and of cars. You look at the color of each image and the apparent size of the vehicle in it and plot the data on a graph. Given the following points (pink dots are bikes and blue are cars), how would you choose to separate this data?

![Image](https://video.udacity-data.com/topher/2018/March/5ab58682_screen-shot-2018-03-23-at-3.57.38-pm/screen-shot-2018-03-23-at-3.57.38-pm.png)

Pink and blue dots representing the size and color of bikes (pink) and cars (blue). The size is on the x-axis and the color on the left axis. Cars tend to be larger than bikes, but both come in a variety of colors.

![Image](https://video.udacity-data.com/topher/2018/March/5ab586c8_screen-shot-2018-03-23-at-3.59.04-pm/screen-shot-2018-03-23-at-3.59.04-pm.png)
Answer: D

### Layers of Separation
What if the data looked like this?

![Image](https://video.udacity-data.com/topher/2018/March/5ab587ad_screen-shot-2018-03-23-at-4.02.58-pm/screen-shot-2018-03-23-at-4.02.58-pm.png)

Pink (bike) and blue (car) dots on a similar size-color graph. This time, the blue dots are collected in the top right quadrant of the graph, indicating that cars come in a more limited color palette.

You could combine two different lines of separation! You could even plot a curved line to separate the blue dots from the pink, and this is what machine learning learns to do — to choose the best algorithm to separate any given data.

![Image](https://video.udacity-data.com/topher/2018/March/5ab5880d_screen-shot-2018-03-23-at-4.04.35-pm/screen-shot-2018-03-23-at-4.04.35-pm.png)

Two, slightly-angled lines, each of which divides the data into two groups.

![Image](https://video.udacity-data.com/topher/2018/March/5ab5884a_screen-shot-2018-03-23-at-4.05.32-pm/screen-shot-2018-03-23-at-4.05.32-pm.png)

Both lines, combined, clearly separate the car and bike data!

### Images as Numerical Data
Every pixel in an image is just a numerical value and, we can also change these pixel values. We can multiply every single one by a scalar to change how bright the image is, we can shift each pixel value to the right, and many more operations!

**Treating images as grids of numbers is the basis for many image processing techniques.**

Most color and shape transformations are done just by mathematically operating on an image and changing it pixel-by-pixel.

### Color Images
Color images are interpreted as 3D cubes of values with width, height, and depth!

The depth is the number of colors. Most color images can be represented by combinations of only 3 colors: red, green, and blue values; these are known as RGB images. And for RGB images, the depth is 3!

It’s helpful to think of the depth as three stacked, 2D color layers. One layer is Red, one Green, and one Blue. Together they create a complete color image.

![Image](https://video.udacity-data.com/topher/2017/December/5a386e3b_screen-shot-2017-12-18-at-5.41.01-pm/screen-shot-2017-12-18-at-5.41.01-pm.png)

RGB layers of a car image.

#### Importance of Color
In general, when you think of a classification challenge, like identifying lane lines or cars or people, you can decide whether color information and color images are useful by thinking about your own vision.

If the identification problem is easier in color for us humans, it’s likely easier for an algorithm to see color images too!

#### Why BGR instead of RGB?
OpenCV reads in images in BGR format (instead of RGB) because when OpenCV was first being developed, BGR color format was popular among camera manufacturers and image software providers. The red channel was considered one of the least important color channels, so was listed last, and many bitmaps use BGR format for image storage. However, now the standard has changed and most image software and cameras use RGB format, which is why, in these examples, it's good practice to initially convert BGR images to RGB before analyzing or manipulating them.

**Changing Color Spaces**  
To change color spaces, we used OpenCV's cvtColor function, whose documentation is [here](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html).

### Why do we need labels?
You can tell if an image is night or day, but a computer cannot unless we tell it explicitly with a label!

This becomes especially important when we are testing the accuracy of a classification model.

A classifier takes in an image as input and should output a predicted_label that tells us the predicted class of that image. Now, when we load in data, like you’ve seen, we load in what are called the true_labels which are the correct labels for the image.

To check the accuracy of a classification model, we compare the predicted and true labels. If the true and predicted labels match, then we’ve classified the image correctly! Sometimes the labels do not match, which means we’ve misclassified an image.

![Image](https://video.udacity-data.com/topher/2017/December/5a38914e_screen-shot-2017-12-18-at-8.09.57-pm/screen-shot-2017-12-18-at-8.09.57-pm.png)

A misclassified image example. The true_label is "day" and the predicted_label is "night".

### Accuracy
After looking at many images, the accuracy of a classifier is defined as the number of correctly classified images (for which the predicted_label matches the true label) divided by the total number of images. So, say we tried to classify 100 images total, and we correctly classified 81 of them. We’d have 0.81 or 81% accuracy!

We can tell a computer to check the accuracy of a classifier only when we have these predicted and true labels to compare. We can also learn from any mistakes the classifier makes, as we’ll see later in this lesson.

### Numerical labels
It’s good practice to use numerical labels instead of strings or categorical labels. They're easier to track and compare. So, for day and night classification, it's binary class example, instead of "day" and "night" labels we’ll use the numerical labels: 0 for night and 1 for day.

### Distinguishing and Measurable Traits
When you approach a classification challenge, you may ask yourself: how can I tell these images apart? What traits do these images have that differentiate them, and how can I write code to represent their differences? Adding on to that, how can I ignore irrelevant or overly similar parts of these images?

You may have thought about a number of distinguishing features: day images are much brighter, generally, than night images. Night images also have these really bright small spots, so the brightness over the whole image varies a lot more than the day images. There is a lot more of a gray/blue color palette in the day images.

There are lots of measurable traits that distinguish these images, and these measurable traits are referred to as features.

A feature a measurable component of an image or object that is, ideally, unique and recognizable under varying conditions - like under varying light or camera angle. And we’ll learn more about features soon.

### Standardizing and Pre-processing
But we’re getting ahead of ourselves! To extract features from any image, we have to pre-process and standardize them!

Next we’ll take a look at the standardization steps we should take before we can consistently extract features.

### Numerical vs. Categorical
Let's learn a little more about labels. After visualizing the image data, you'll have seen that each image has an attached label: "day" or "night," and these are known as **categorical values**.

Categorical values are typically text values that represent various traits about an image. A couple examples are:

- An "animal" variable with the values: "cat," "tiger," "hippopotamus," and "dog."
- A "color" variable with the values: "red," "green," and "blue."

Each value represents a different category, and most collected data is labeled in this way!

These labels are descriptive for us, but may be inefficient for a classification task. Many machine learning algorithms do not use categorical data; they require that all output be numerical. Numbers are easily compared and stored in memory, and for this reason, we often have to convert categorical values into numerical labels. There are two main approaches that you'll come across:

1. Integer encoding
2. One hot-encoding

#### Integer Encoding
Integer encoding means to assign each category value an integer value. So, day = 1 and night = 0. This is a nice way to separate binary data, and it's what we'll do for our day and night images.

#### One-hot Encoding
One-hot encoding is often used when there are more than 2 values to separate. A one-hot label is a 1D list that's the length of the number of classes. Say we are looking at the animal variable with the values: "cat," "tiger," "hippopotamus," and "dog." There are 4 classes in this category and so our one-hot labels will be a list of length four. The list will be all 0's and one 1; the 1 indicates which class a certain image is.

For example, since we have four classes (cat, tiger, hippopotamus, and dog), we can make a list in that order: `[cat value, tiger value, hippopotamus value, dog value]`. In general, order does not matter.

If we have an image and it's one-hot label is `[0, 1, 0, 0]`, what does that indicate?

In order of [cat value, tiger value, hippopotamus value, dog value], that label indicates that it's an image of a tiger! Let's do one more example, what about the label `[0, 0, 0, 1]`?

### Classification Task
Let’s now complete our day and night classifier. After we extracted the average brightness value, we want to turn this feature into a `predicted_label` that classifies the image. Remember, we want to generate a numerical label, and again, since we have a binary dataset, I’ll create a label that is a 1 if an image is predicted to be day and a 0 for images predicted to be night.

I can create a complete classifier by writing a function that takes in an image, extracts the brightness feature, and then checks if the average brightness is above some threshold X.

If it is, this classifier returns a 1 (day), and if it’s not, this classifier returns a 0 (night)!

Next, you'll take a look at this notebook and get a chance to tweak the threshold parameter. Then, when you're able to generate predicted labels, you can compare them to the true labels, and check the accuracy of our model!

### Evaluation Metrics
#### Accuracy
The accuracy of a classification model is found by comparing predicted and true labels. For any given image, if the `predicted_label` matches the `true_label`, then this is a correctly classified image, if not, it is misclassified.

The accuracy is given by the number of correctly classified images divided by the total number of images. We’ll test this classification model on new images, this is called a test set of data.

#### Test Data
Test data is previously unseen image data. The data you have seen, and that you used to help build a classifier is called training data, which we've been referring to. The idea in creating these two sets is to have one set that you can analyze and learn from (training), and one that you can get a sense of how your classifier might work in a real-world, general scenario. You could imagine going through each image in the training set and creating a classifier that can classify all of these training images correctly, but, you actually want to build a classifier that **recognizes general patterns** in data, so that when it is faced with a real-world scenario, it will still work!

So, we use a new, test set of data to see how a classification model might work in the real-world and to determine the accuracy of the model.

#### Misclassified Images
In this and most classification examples, there are a few misclassified images in the test set. To see how to improve, it’s useful to take a look at these misclassified images; look at what they were mistakenly labeled as and where your model fails. It will be up to you to look at these images and think about how to improve the classification model!
