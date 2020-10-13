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
