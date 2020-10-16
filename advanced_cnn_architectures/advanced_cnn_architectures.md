# Advanced CNN Architectures
## Overview
[Detailed Explaination Video](https://www.youtube.com/watch?v=_iRqSOsTBQU)

In real world, there's usually more complex objects in the image, and we would even need to detect their distance etc. This section looks at how to deal with more complex tasks using CNN.

## Classification and Localization
[Detailed Explaination Video](https://www.youtube.com/watch?v=vBE5KvvAYzg)
Classifying an object and finding it's location in the image. This typically uses a bounding box using region based CNN such as **Fast R-CNN**.

### Localization
[Detailed Explaination Video](https://www.youtube.com/watch?v=UqNg9d6cKQU)
Finding the position of the object in the image. Typical way of doing this is by drawing bounding boxes where (x, y) is the centre of the image and (h, w) is the height and width of the box. Typically, another fully connected layer is attached to the feature vector to predict bounding box while another is used to predict the class.

#### Bounding Boxes
[Detailed Explaination Video](https://www.youtube.com/watch?v=2YM82c7SaCo)
We typically use cross entropy loss to compute loss in classfication, but when we look at locations, it becomes a regression problem where cross entropy loss is **not applicable**. Instead, we use loss functions such as **mean square error**.

Regression Loss Types:
- L1 Loss: Difference in points
- MSE Loss: (Mean Square Error
- Smooth L1 Loss: Combine both L1 and MSE

#### Beyond Bounding Boxes
To predict bounding boxes, we train a model to take an image as input and output coordinate values: (x, y, w, h). This kind of model can be extended to any problem that has coordinate values as outputs! One such example is human pose estimation.

![Image](https://video.udacity-data.com/topher/2018/May/5aeb6ac2_screen-shot-2018-05-03-at-1.01.46-pm/screen-shot-2018-05-03-at-1.01.46-pm.png)

Huan pose estimation points.

In the above example, we see that the pose of a human body can be estimated by tracking 14 points along the joints of a body.

### Weighted Loss Functions
You may be wondering: how can we train a network with two different outputs (a class and a bounding box) and different losses for those outputs?

We know that, in this case, we use categorical cross entropy to calculate the loss for our predicted and true classes, and we use a regression loss (something like smooth L1 loss) to compare predicted and true bounding boxes. But, we have to train our whole network using one loss, so how can we combine these?

There are a couple of ways to train on multiple loss functions, and in practice, we often use a weighted sum of classification and regression losses (ex.` 0.5*cross_entropy_loss + 0.5*L1_loss`); the result is a single error value with which we can do backpropagation. This does introduce a hyperparameter: the loss weights. We want to weight each loss so that these losses are balanced and combined effectively, and in research we see that another regularization term is often introduced to help decide on the weight values that best combine these losses.

## Region Proposals
[Detailed Explaination Video](https://www.youtube.com/watch?v=HLwpr7h3rPY)

When there are multiple objects in the image, we split the objects into different bounding boxes. The real challenge is that we do not know how many boxes we need, so to detect multiple objects, we produce multile bounding boxes for each region, then you classify each region. One of the technqiue to produce all the bounding boxes are to use sliding window, but this is very inefficient as many of the boxes will not contain any objects. A more efficient technqiue would be to use R-CNN

### R-CNN (Region CNN)
[Detailed Explaination Video](https://www.youtube.com/watch?v=EchapZJMTYU)

Region proposals produce a set of bounding boxes where objects are more likely to be in. Region proposal algorithm produces a set of Region of Interest (ROI). These ROIs are then reduce to a standard size and inputed into CNN. 

#### R-CNN Outputs
The R-CNN is the least sophisticated region-based architecture, but it is the basis for understanding how multiple object recognition algorithms work! It outputs a class score and bounding box coordinates for every input RoI.

An R-CNN feeds an image into a CNN with regions of interest (RoI’s) already identified. Since these RoI’s are of varying sizes, they often need to be **warped to be a standard size**, since CNN’s typically expect a consistent, square image size as input. After RoI's are warped, the R-CNN architecture, processes these regions one by one and, for each image, produces 1. a class label and 2. a bounding box (that may act as a slight correction to the input region).

R-CNN produces bounding box coordinates to reduce localization errors; so a region comes in, but it may not perfectly surround a given object and the output coordinates `(x,y,w,h)` aim to perfectly localize an object in a given region.
R-CNN, unlike other models, does not explicitly produce a confidence score that indicates whether an object is in a region, instead it cleverly produces a set of class scores for which one class is "background". This ends up serving a similar purpose, for example, if the class score for a region is `Pbackground = 0.10`, it likely contains an object, but if it's `Pbackground = 0.90`, then the region probably doesn't contain an object.

### Fast R-CNN
[Detailed Explaination Video](https://www.youtube.com/watch?v=6FOBZ9OgWlY)

Instead of processing each ROI individual through the classficication CNN. This architecture runs the image into the classfication CNN only once. Fast R-CNN employs several innovations to improve training and testing speed while also increasing detection accuracy. Instead of feeding the region proposals to the CNN, we feed the input image to the CNN to generate a convolutional feature map. From the convolutional feature map, we identify the region of proposals and warp them into squares and by using a RoI pooling layer we reshape them into a fixed size so that it can be fed into a fully connected layer. From the RoI feature vector, we use a softmax layer to predict the class of the proposed region and also the offset values for the bounding box.

#### RoI Pooling
To warp regions of interest into a consistent size for further analysis, some networks use RoI pooling. RoI pooling is an additional layer in our network that takes in a rectangular region of any size, performs a maxpooling operation on that region in pieces such that the output is a fixed shape. Below is an example of a region with some pixel values being broken up into pieces which pooling will be applied to; a section with the values:

```
[[0.85, 0.34, 0.76],
 [0.32, 0.74, 0.21]]
 ```
Will become a single max value after pooling: 0.85. After applying this to an image in these pieces, you can see how any rectangular region can be forced into a smaller, square representation.

![Image](https://video.udacity-data.com/topher/2018/May/5aeb9cc4_screen-shot-2018-05-03-at-4.34.25-pm/screen-shot-2018-05-03-at-4.34.25-pm.png)

An example of pooling sections, credit to this informational resource on RoI pooling [by Tomasz Grel].

You can see the complete process from input image to region to reduced, maxpooled region, below.

![Image](https://video.udacity-data.com/topher/2018/May/5aeb9dc6_roi-pooling-gif/roi-pooling-gif.gif)

#### Speed
Fast R-CNN is about 10 times as fast to train as an R-CNN because it only creates convolutional layers once for a given image and then performs further analysis on the layer. Fast R-CNN also takes a shorter time to test on a new image! It’s test time is dominated by the time it takes to create region proposals.

### Faster R-CNN
Faster R-CNN has a lower time to form region proposals. It takes an input image, and puts into CNN until a certain convolution layer. This time it uses the produced feature map as input into a separate region proposal network. It produce its own feature in the region proposal network. For each ROI, it checks if it contains an object and perform classification if it does. It eliminates the non-object ROI first to speed up the process. 

#### Region Proposal Network
You may be wondering: how exactly are the RoI's generated in the region proposal portion of the Faster R-CNN architecture?

The region proposal network (RPN) works in Faster R-CNN in a way that is similar to YOLO object detection, which you'll learn about in the next lesson. The RPN looks at the output of the last convolutional layer, a produced feature map, and takes a sliding window approach to possible-object detection. It slides a small (typically 3x3) window over the feature map, then for each window the RPN:

1. Uses a set of defined anchor boxes, which are boxes of a defined aspect ratio (wide and short or tall and thin, for example) to generate multiple possible RoI's, each of these is considered a region proposal.
2. For each proposal, this network produces a probability, Pc, that classifies the region as an object (or not) and a set of bounding box coordinates for that object.
3. Regions with too low a probability of being an object, say Pc < 0.5, are discarded.

##### Training the Region Proposal Network
Since, in this case, there are no ground truth regions, how do you train the region proposal network?

The idea is, for any region, you can check to see if it overlaps with any of the ground truth objects. That is, for a region, if we classify that region as an object or not-object, which class will it fall into? For a region proposal that does cover some portion of an object, we should say that there is a high probability that this region has an object init and that region should be kept; if the likelihood of an object being in a region is too low, that region should be discarded.

I'd recommend [this blog post](https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9) if you'd like to learn more about region selection.

#### Speed Bottleneck
Now, for all of these networks including Faster R-CNN, we've aimed to improve the speed of our object detection models by reducing the time it takes to generate and decide on region proposals. 
