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

## Feedforward Neural Network

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

## Backpropagation
### Backpropagation Theory
Since partial derivatives are the key mathematical concept used in backpropagation, it's important that you feel confident in your ability to calculate them. Once you know how to calculate basic derivatives, calculating partial derivatives is easy to understand.

In the **backpropagation** process we minimize the network error slightly with each iteration, by adjusting the weights. The following video will help you understand the mathematical process we use for computing these adjustments.

[Detailed Explaination Video](https://www.youtube.com/watch?v=Xlgd8I3TWUg)

### Overfitting
[Detailed Explaination Video](https://www.youtube.com/watch?v=rmBLnVbFfFY)

Overfitting means the noise of the training set is also mapped onto the model. Generally the two apprach to stop this is to
1. Stop training process early - One way to know when to stop is to use a validation from training set. However, drawback is that this decrease training set size.
2. Regularization - use Dropout to randomly throw away some results.

## RNN temporal depdencies
[Detailed Explaination Video](https://www.youtube.com/watch?v=ofbnDxGSUcg)

RNNs are based on the same principles as those behind FFNNs, which is why we spent so much time reminding ourselves of the feedforward and backpropagation steps that are used in the training phase.

There are two main differences between FFNNs and RNNs. The Recurrent Neural Network uses:

- **sequences** as inputs in the training phase, and
- **memory** elements

Memory is defined as the output of hidden layer neurons, which will serve as additional input to the network during next training step.

The basic three layer neural network with feedback that serve as memory inputs is called the **Elman Network** and is depicted in the following picture:

![Image](https://video.udacity-data.com/topher/2017/November/5a00d738_screen-shot-2017-11-06-at-1.40.14-pm/screen-shot-2017-11-06-at-1.40.14-pm.png)

[RNN folded](https://www.youtube.com/watch?v=wsif3p5t7CI)

As we've see, in FFNN the output at any time t, is a function of the current input and the weights. In RNNs, our output at time t, depends not only on the current input and the weight, but also on previous inputs. 

This is the RNN folded model:

![Image](https://video.udacity-data.com/topher/2017/November/5a00ddac_screen-shot-2017-11-06-at-2.09.07-pm/screen-shot-2017-11-06-at-2.09.07-pm.png)

The RNN folded model

In this picture, x¯ represents the input vector, y¯ represents the output vector and s¯ denotes the state vector. 

W_x is the weight matrix connecting the inputs to the state layer.

W_y is the weight matrix connecting the state layer to the output layer.

W_s represents the weight matrix connecting the state from the previous timestep to the state in the current timestep.

The model can also be "unfolded in time". The **unfolded model** is usually what we use when working with RNNs.

![Image](https://video.udacity-data.com/topher/2017/November/5a00e48f_screen-shot-2017-11-06-at-2.38.51-pm/screen-shot-2017-11-06-at-2.38.51-pm.png)

The RNN unfolded model

In both the folded and unfolded models shown above the following notation is used:

x¯ represents the input vector, \bar{y} 

y¯ represents the output vector and \bar{s} 

s¯ represents the state vector.

W_x is the weight matrix connecting the inputs to the state layer.

W_y is the weight matrix connecting the state layer to the output layer.

W_s represents the weight matrix connecting the state from the previous timestep to the state in the current timestep.

In RNNs the state layer depended on the current inputs, their corresponding weights, the activation function and also on the previous state:

![Image](https://video.udacity-data.com/topher/2017/November/5a00e614_screen-shot-2017-11-06-at-2.45.22-pm/screen-shot-2017-11-06-at-2.45.22-pm.png)

The output vector is calculated exactly the same as in FFNNs. It can be a linear combination of the inputs to each output node with the corresponding weight matrix W_y, or a softmax function of the same linear combination.

[RNN unfolded model](https://www.youtube.com/watch?v=xLIA_PTWXog)

### RNN Example
[Sequence Detector](https://www.youtube.com/watch?v=MDLk3fhpTx0)

### Backpropagation Through Time
[Part 1](https://www.youtube.com/watch?v=eE2L3-2wKac). 
[Part 2](https://www.youtube.com/watch?v=bUU9BEQw0IA). 
[Part 3](https://www.youtube.com/watch?v=uBy_eIJDD1M). 

We are now ready to understand how to train the RNN.

When we train RNNs we also use backpropagation, but with a conceptual change. The process is similar to that in the FFNN, with the exception that we need to consider previous time steps, as the system has memory. This process is called Backpropagation Through Time (BPTT) and will be the topic of the next three videos.

As always, don't forget to take notes.
In the following videos we will use the Loss Function for our error. The Loss Function is the square of the difference between the desired and the calculated outputs. There are variations to the Loss Function, for example, factoring it with a scalar. In the backpropagation example we used a factoring scalar of 1/2 for calculation convenience.

As described previously, the two most commonly used are the Mean Squared Error (MSE) (usually used in regression problems) and the cross entropy (usually used in classification problems).

Here, we are using a variation of the MSE.

In **BPTT** we train the network at timestep t as well as take into account all of the previous timesteps.

The easiest way to explain the idea is to simply jump into an example.

In this example we will focus on the BPTT process for time step t=3. You will see that in order to adjust all three weight matrices, W_x, W_s and W_y, we need to consider timestep 3 as well as timestep 2 and timestep 1.

As we are focusing on timestep t=3, the Loss function will be: E_3 =(d¯_3 − y¯_3)^2
 
![Image](https://video.udacity-data.com/topher/2017/November/5a1c8721_screen-shot-2017-11-27-at-1.43.36-pm/screen-shot-2017-11-27-at-1.43.36-pm.png)

The Folded Model at Timestep 3

To update each weight matrix, we need to find the partial derivatives of the Loss Function at time 3, as a function of all of the weight matrices. We will modify each matrix using gradient descent while considering the previous timesteps.

![Image](https://video.udacity-data.com/topher/2017/November/5a1c87d9_screen-shot-2017-11-27-at-1.46.43-pm/screen-shot-2017-11-27-at-1.46.43-pm.png)

Gradient Considerations in the Folded Model

We will now unfold the model. You will see that unfolding the model in time is very helpful in visualizing the number of steps (translated into multiplication) needed in the Backpropagation Through Time process. These multiplications stem from the chain rule and are easily visualized using this model.

In this video we will understand how to use Backpropagation Through Time (BPTT) when adjusting two weight matrices:

W_y - the weight matrix connecting the state the output
W_s - the weight matrix connecting one state to the next state

he unfolded model can be very helpful in visualizing the BPTT process.

![Image](https://video.udacity-data.com/topher/2017/November/5a1c8a95_screen-shot-2017-11-27-at-1.58.01-pm/screen-shot-2017-11-27-at-1.58.01-pm.png)

The Unfolded Model at timestep 3

Gradient calculations needed to adjust W_y

The partial derivative of the Loss Function with respect to W_y is found by a simple one step chain rule: (Note that in this case we do not need to use BPTT. Visualization of the calculations path can be found in the video).

![Image](https://video.udacity-data.com/topher/2017/November/5a14b8fb_screen-shot-2017-11-21-at-3.38.11-pm/screen-shot-2017-11-21-at-3.38.11-pm.png)

Equation 36

Generally speaking, we can consider multiple timesteps back, and not only 3 as in this example. For an arbitrary timestep N, the gradient calculation needed for adjusting W_y, is:
![Image](https://video.udacity-data.com/topher/2017/November/5a14c338_screen-shot-2017-11-21-at-4.21.41-pm/screen-shot-2017-11-21-at-4.21.41-pm.png)

Equation 37

**Gradient calculations needed to adjust W_s**

We still need to adjust W_s the weight matrix connecting one state to the next and W_x the weight matrix connecting the input to the state. We will arbitrarily start with W_s.

To understand the BPTT process, we can simplify the unfolded model. We will focus on the contributions of W_s to the output, the following way:

![Image](https://video.udacity-data.com/topher/2017/November/5a1c8b04_screen-shot-2017-11-27-at-2.00.15-pm/screen-shot-2017-11-27-at-2.00.15-pm.png)

When calculating the partial derivative of the Loss Function with respect to W_s, we need to consider all of the states contributing to the output. In the case of this example it will be states s3¯ which depends on its predecessor s2¯ which depends on its predecessor s1¯, the first state.

In BPTT we will take into account every gradient stemming from each state, **accumulating** all of these contributions.

- At timestep t=3, the contribution to the gradient stemming from s3¯ is the following : (Notice the use of the chain rule here)

![Image](https://video.udacity-data.com/topher/2017/November/5a14ba05_screen-shot-2017-11-21-at-3.42.29-pm/screen-shot-2017-11-21-at-3.42.29-pm.png)

Equation 38

- At timestep t=3, the contribution to the gradient stemming from s2¯ is the following : (Notice how the equation, derived by the chain rule, considers the contribution of s2¯ to s3¯. 

![Image](https://video.udacity-data.com/topher/2017/November/5a14ba6b_screen-shot-2017-11-21-at-3.44.15-pm/screen-shot-2017-11-21-at-3.44.15-pm.png)

Equation 39

At timestep t=3, the contribution to the gradient stemming from s1¯ is the following : 

![Image](https://video.udacity-data.com/topher/2017/November/5a14bac7_screen-shot-2017-11-21-at-3.45.50-pm/screen-shot-2017-11-21-at-3.45.50-pm.png)

Equation 40

After considering the contributions from all three states, we will accumulate them to find the final gradient calculation.

The following equation is the gradient contributing to the adjustment of W_s using Backpropagation Through Time:

![Image](https://video.udacity-data.com/topher/2017/November/5a14bac7_screen-shot-2017-11-21-at-3.45.50-pm/screen-shot-2017-11-21-at-3.45.50-pm.png)

Equation 41

In this example we had 3 time steps to consider, therefore we accumulated three partial derivative calculations. Generally speaking, we can consider multiple timesteps back. If you look closely at the three components of equation 41, you will notice a pattern. You will find that as we propagate a step back, we have an additional partial derivatives to consider in the chain rule. Mathematically this can be easily written in the following general equation for adjusting W_s using BPTT:

![Image](https://video.udacity-data.com/topher/2017/November/5a14c286_screen-shot-2017-11-21-at-4.17.35-pm/screen-shot-2017-11-21-at-4.17.35-pm.png)

Equation 42

Notice that Equation 6 considers a general setting of N steps back. As mentioned in this lesson, capturing relationships that span more than 8 to 10 steps back is practically impossible due to the vanishing gradient problem. We will talk about a solution to this problem in our LSTM section coming up soon.

We still need to adjust W_x, the weight matrix connecting the input to the state.

The following equation is the gradient contributing to the adjustment of W_x using Backpropagation Through Time:

![Image](https://video.udacity-data.com/topher/2017/November/5a14c18c_screen-shot-2017-11-21-at-4.14.45-pm/screen-shot-2017-11-21-at-4.14.45-pm.png)

As mentioned before, in this example we had 3 time steps to consider, therefore we accumulated three partial derivative calculations. Generally speaking, we can consider multiple timesteps back. If you look closely at equations 1, 2 and 3, you will notice a pattern again. You will find that as we propagate a step back, we have an additional partial derivatives to consider in the chain rule. Mathematically this can be easily written in the following general equation for adjusting W_x using BPTT:

![Image](https://video.udacity-data.com/topher/2017/November/5a14c244_screen-shot-2017-11-21-at-4.17.19-pm/screen-shot-2017-11-21-at-4.17.19-pm.png)

## RNN Summary
[Detailed Explaination Video](https://www.youtube.com/watch?v=nXP0oGGRrO8)

When training RNNs using BPTT, we can choose to use mini-batches, where we update the weights in batches periodically (as opposed to once every inputs sample). We calculate the gradient for each step but do not update the weights right away. Instead, we update the weights once every fixed number of steps. This helps reduce the complexity of the training process and helps remove noise from the weight updates.

The following is the equation used for Mini-Batch Training Using Gradient Descent: delta_ij represents the gradient calculated once every inputs sample and M represents the number of gradients we accumulate in the process).

![Image](https://video.udacity-data.com/topher/2017/November/5a05088c_screen-shot-2017-11-09-at-6.01.16-pm/screen-shot-2017-11-09-at-6.01.16-pm.png)

Equation 61

If we backpropagate more than ~10 timesteps, the gradient will become too small. This phenomena is known as the **vanishing gradient** problem where the contribution of information decays geometrically over time. Therefore temporal dependencies that span many time steps will effectively be discarded by the network. **Long Short-Term Memory (LSTM)** cells were designed to specifically solve this problem.

In RNNs we can also have the opposite problem, called the **exploding gradient problem**, in which the value of the gradient grows uncontrollably. A simple solution for the exploding gradient problem is **Gradient Clipping**.
