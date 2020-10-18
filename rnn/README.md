# Recurrent Neural Networks (RNN)

[Detailed Explaination Video](https://www.youtube.com/watch?v=AIQEqg6F38A)

So far, we've been looking at convolutional neural networks and models that allows us to analyze the spatial information in a given input image. CNN's excel in tasks that rely on finding spatial and visible patterns in training data.

In this and the next couple lessons, we'll be reviewing RNN's or recurrent neural networks. These networks give us a way to incorporate memory into our neural networks, and will be critical in analyzing sequential data. RNN's are most often associated with text processing and text generation because of the way sentences are structured as a sequence of words, but they are also useful in a number of computer vision applications, as well!

## RNN's in Computer Vision
At the end of this lesson, you will be tasked with creating an automatic image captioning model that takes in an image as input and outputs a sequence of words, describing that image. Image captions are used to create accessible content and in a number of other cases where one may want to read about the contents of an image. This model will include a CNN component for finding spatial patterns in the input image and and RNN component that will be responsible for generative descriptive text!

RNN's are also sometimes used to analyze sequences of images; this can be useful in captioning video, as well as video classification, gesture recognition, and object tracking; all of these tasks see as input a sequence of image frames.

## Sketch RNN
One of my favorite use cases for RNN's in computer vision tasks is in generating drawings. Sketch RNN ([demo here](https://magenta.tensorflow.org/assets/sketch_rnn_demo/index.html)) is a program that learns to complete a drawing, once you give it something (a line or circle, etc.) to start!

![Image](https://video.udacity-data.com/topher/2018/May/5af0dfc7_screen-shot-2018-05-07-at-4.20.50-pm/screen-shot-2018-05-07-at-4.20.50-pm.png)

Sketch RNN example output. Left, Mona Lisa. Right, pineapple.

It's interesting to think of drawing as a sequential act, but it is! This network takes a starting line or squiggle and then, having trained on a number of types of sketches, does it's best to complete the drawing based on your input squiggle.

Next, you'll learn all about how RNN's are structured and how they can be trained! This section is taught by Ortal, who has a PhD in Computer Engineering and has been a professor and researcher in the fields of applied cryptography and embedded systems.

## History

[Detailed Explaination Video](https://www.youtube.com/watch?v=HbxAnYUfRnc)

As mentioned in this video, RNNs have a key flaw, as capturing relationships that span more than 8 or 10 steps back is practically impossible. This flaw stems from the "**vanishing gradient**" problem in which the contribution of information decays geometrically over time.

### What does this mean?

As you may recall, while training our network we use **backpropagation**. In the backpropagation process we adjust our weight matrices with the use of a **gradient**. In the process, gradients are calculated by continuous multiplications of derivatives. The value of these derivatives may be so small, that these continuous multiplications may cause the gradient to practically "vanish".

**LSTM** is one option to overcome the Vanishing Gradient problem in RNNs.

## Feedforward Neural Network - A Reminder

![Image](https://video.udacity-data.com/topher/2017/October/59de877f_screen-shot-2017-10-11-at-2.04.14-pm/screen-shot-2017-10-11-at-2.04.14-pm.png)

The mathematical calculations needed for training RNN systems are fascinating. To deeply understand the process, we first need to feel confident with the vanilla FFNN system. We need to thoroughly understand the feedforward process, as well as the backpropagation process used in the training phases of such systems. The next few videos will cover these topics, which you are already familiar with. We will address the feedforward process as well as backpropagation, using specific examples. These examples will serve as extra content to help further understand RNNs later in this lesson.

The following couple of videos will give you a brief overview of the Feedforward Neural Network (FFNN).

[FFNN](https://www.youtube.com/watch?v=_vrp2lZjXf0)

OK, you can take a small break now. We will continue with FFNN when you come back!

[FFNN propagation](https://www.youtube.com/watch?v=FfPjaGcZODc)

As mentioned before, when working with neural networks we have 2 primary phases:

**Training** and **Evaluation**.

During the **training** phase, we take the data set (also called the training set), which includes many pairs of inputs and their corresponding targets (outputs). Our goal is to find a set of weights that would best map the inputs to the desired outputs. In the **evaluation** phase, we use the network that was created in the training phase, apply our new inputs and expect to obtain the desired outputs.

The training phase will include two steps:

**Feedforward** and **Backpropagation**

We will repeat these steps as many times as we need until we decide that our system has reached the best set of weights, giving us the best possible outputs.

## Feedforward

In this section we will look closely at the math behind the feedforward process. With the use of basic Linear Algebra tools, these calculations are pretty simple!

Assuming that we have a single hidden layer, we will need two steps in our calculations. The first will be calculating the value of the hidden states and the latter will be calculating the value of the outputs.

![Image](https://video.udacity-data.com/topher/2017/October/59f39785_screen-shot-2017-10-27-at-1.29.13-pm/screen-shot-2017-10-27-at-1.29.13-pm.png)

Notice that both the hidden layer and the output layer are displayed as vectors, as they are both represented by more than a single neuron.

Our first [video](https://www.youtube.com/watch?v=4rCfnWbx8-0) will help you understand the first step - **Calculating the value of the hidden states**.

As you saw in the video above, vector h' of the hidden layer will be calculated by multiplying the input vector with the weight matrix.

Using vector by matrix multiplication, we can look at this computation the following way:

![Image](https://video.udacity-data.com/topher/2017/December/5a26f9f2_screen-shot-2017-12-05-at-11.55.58-am/screen-shot-2017-12-05-at-11.55.58-am.png)

Equation 1

After finding h', we need an activation function (Φ) to finalize the computation of the hidden layer's values. This activation function can be a Hyperbolic Tangent, a Sigmoid or a ReLU function. We can use the following two equations to express the final hidden vector h

Since Wij represents the weight component in the weight matrix, connecting neuron i from the input to neuron j in the hidden layer, we can also write these calculations in the following way: (notice that in this example we have n inputs and only 3 hidden neurons)

![Image](https://video.udacity-data.com/topher/2017/October/59f3ddb6_screen-shot-2017-10-27-at-6.29.49-pm/screen-shot-2017-10-27-at-6.29.49-pm.png)

Equation 2

This next [video](https://www.youtube.com/watch?v=kTYbTVh1d0k) will help you understand the second step - **Calculating the values of the Outputs**.

As you've seen in the video above, the process of calculating the output vector is mathematically similar to that of calculating the vector of the hidden layer. We use, again, a vector by matrix multiplication, which can be followed by an activation function. The vector is the newly calculated hidden layer and the matrix is the one connecting the hidden layer to the output.

![Image](https://video.udacity-data.com/topher/2017/October/59f767b8_screen-shot-2017-10-30-at-10.54.50-am/screen-shot-2017-10-30-at-10.54.50-am.png)

Essentially, each new layer in an neural network is calculated by a vector by matrix multiplication, where the vector represents the inputs to the new layer and the matrix is the one connecting these new inputs to the next layer.

In our example, the input vector is h and the matrix is W^2, therefore y=h * W^2 . In some applications it can be beneficial to use a softmax function (if we want all output values to be between zero and 1, and their sum to be 1).

![Image](https://video.udacity-data.com/topher/2017/October/59f775fc_screen-shot-2017-10-30-at-11.56.27-am/screen-shot-2017-10-30-at-11.56.27-am.png)

Equation 3

The two error functions that are most commonly used are the Mean Squared Error (MSE) (usually used in regression problems) and the cross entropy (usually used in classification problems).

In the above calculations we used a variation of the MSE.
