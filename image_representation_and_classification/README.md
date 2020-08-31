# Image Representation and Classification
## Computer Vision Pipeline
A computer vision pipeline is a series of steps that most computer vision applications will go through. Many vision applications start off by acquiring images and data, then processing that data, performing some analysis and recognition steps, then finally performing an action. The general pipeline is pictured below!

**General computer vision processing pipeline**
![Image](https://video.udacity-data.com/topher/2017/March/58d03bda_screen-shot-2017-03-13-at-12.36.54-pm/screen-shot-2017-03-13-at-12.36.54-pm.png)

Now, let's take a look at a specific example of a pipeline applied to facial expression recognition.

**Facial recognition pipeline**
![Image](https://video.udacity-data.com/topher/2018/April/5ade67ef_screen-shot-2018-04-23-at-4.10.20-pm/screen-shot-2018-04-23-at-4.10.20-pm.png)

## Standardizing Data
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
