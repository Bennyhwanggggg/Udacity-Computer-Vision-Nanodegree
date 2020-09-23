# CNN Layers and Feature Visualisation
## CNN Structure
A classification CNN takes in an input image and outputs a distribution of class scores, from which we can find the most likely class for a given image. As you go through this lesson, you may find it useful to consult this blog post, which describes the image classification pipeline and the layers that make up a CNN.

![Image](https://video.udacity-data.com/topher/2018/April/5adf80db_screen-shot-2018-04-24-at-12.08.02-pm/screen-shot-2018-04-24-at-12.08.02-pm.png)

A classification CNN structure.

## CNN Layers
The CNN itself is comprised of a number of layers; layers that extract features from input images, reduce the dimensionality of the input, and eventually produce class scores. In this lesson, we'll go over all of these different layers, so that you know how to define and train a complete CNN!

![Image](https://video.udacity-data.com/topher/2018/April/5adf8153_screen-shot-2018-04-24-at-12.07.41-pm/screen-shot-2018-04-24-at-12.07.41-pm.png)

Detailed layers that make up a classification CNN.

## Pre-processing
Look at the steps below to see how pre-processing plays a major role in the creation of this dataset.

![Image](https://video.udacity-data.com/topher/2018/April/5adfc2fe_screen-shot-2018-04-24-at-4.49.51-pm/screen-shot-2018-04-24-at-4.49.51-pm.png)

Pre-processing steps for FashionMNIST data creation.

## Convolutional Neural Networks (CNN's)
The type of deep neural network that is most powerful in image processing tasks, such as sorting images into groups, is called a Convolutional Neural Network (CNN). CNN's consist of layers that process visual information. A CNN first takes in an input image and then passes it through these layers. There are a few different types of layers, and we'll start by learning about the most commonly used layers: convolutional, pooling, and fully-connected layers.

First, let's take a look at a complete CNN architecture; below is a network called VGG-16, which has been trained to recognize a variety of image classes. It takes in an image as input, and outputs a predicted class for that image. The various layers are labeled and we'll go over each type of layer in this network in the next series of videos.

![Image](https://video.udacity-data.com/topher/2018/April/5ac80056_vgg-16/vgg-16.png)

VGG-16 architecture

### Convolutional Layer
The first layer in this network, that processes the input image directly, is a convolutional layer.

- A convolutional layer takes in an image as input.
- A convolutional layer, as its name suggests, is made of a set of convolutional filters (which you've already seen and programmed).
- Each filter extracts a specific kind of feature, ex. a high-pass filter is often used to detect the edge of an object.
- The output of a given convolutional layer is a set of **feature maps** (also called activation maps), which are filtered versions of an original input image.

### Activation Function
You may also note that the diagram reads "convolution + ReLu," and the **ReLu** stands for Rectified Linear Unit (ReLU) activation function. This activation function is zero when the input x <= 0 and then linear with a slope = 1 when x > 0. ReLu's, and other activation functions, are typically placed after a convolutional layer to slightly transform the output so that it's more efficient to perform backpropagation and effectively train the network.

