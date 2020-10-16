# You Only Look Once (YOLO)
[Detailed Explaination Video](https://www.youtube.com/watch?v=uyefSrHZesY)

Traditional R-CNN method is not fast enough for real time, but YOLO is. YOLO has less error as traditional R-CNN method introduce error for each input.

## How it works?
[Detailed Explaination Video](https://www.youtube.com/watch?v=MyOuuwk0qC4)

YOLO takes a different approach by merging the predicted class and ROI box vector into a single output vector. Therefore, YOLO outputs a vector of
`c1, c2, c3, x, y, w, h]` where each `c` for each class and `(x, y)` for coordinates of the center of bounding box, `w, h` for width and height.

### Sliding Window
[Detailed Explaination Video](https://www.youtube.com/watch?v=8qYqqibIz90)

Sliding window chose a window size and slide it across of the image. For each window, it will output whether an object in there combine with the class, x, y, w and h vector. If no object is detected, the other values are 0. We repeat this process until we go over the whole image. We may also vary the size of the window to capture different size objects. 
However, sliding window is too computationally expensive.

#### A Convolutional Approach to Sliding Windows
Let’s assume we have a 16 x 16 x 3 image, like the one shown below. This means the image has a size of 16 by 16 pixels and has 3 channels, corresponding to RGB.

![Image](https://video.udacity-data.com/topher/2018/May/5aef7ae3_diapositiva1/diapositiva1.png)

Let’s now select a window size of 10 x 10 pixels as shown below:

![Image](https://video.udacity-data.com/topher/2018/May/5aef7b30_diapositiva2/diapositiva2.png)

If we use a stride of 2 pixels, it will take 16 windows to cover the entire image, as we can see below.

![Image](https://video.udacity-data.com/topher/2018/May/5aef7bc0_diapositiva3/diapositiva3.png)

In the original Sliding Windows approach, each of these 16 windows will have to be passed individually through a CNN. Let’s assume that CNN has the following architecture:

![Image](https://video.udacity-data.com/topher/2018/May/5aefb1ef_diapositiva4/diapositiva4.jpg)

The CNN takes as input a 10 x 10 x 3 image, then it applies 5, 7 x 7 x 3 filters, then it uses a 2 x 2 Max pooling layer, then is has 128, 2 x 2 x 5 filters, then is has 128, 1 x 1 x 128 filters, and finally it has 8, 1 x 1 x 128 filters that represents a softmax output.

What will happen if we change the input of the above CNN from 10 x 10 x 3, to 16 x 16 x 3? The result is shown below:

![Image](https://video.udacity-data.com/topher/2018/May/5aefb270_diapositiva5-1/diapositiva5-1.png)

As we can see, this CNN architecture is the same as the one shown before except that it takes as input a 16 x 16 x 3 image. The sizes of each layer change because the input image is larger, but the same filters as before have been applied.

If we follow the region of the image that corresponds to the first window through this new CNN, we see that the result is the upper-left corner of the last layer (see image above). Similarly, if we follow the section of the image that corresponds to the second window through this new CNN, we see the corresponding result in the last layer:

![Image](https://video.udacity-data.com/topher/2018/May/5aefb2c0_diapositiva6-1/diapositiva6-1.png)

Likewise, if we follow the section of the image that corresponds to the third window through this new CNN, we see the corresponding result in the last layer, as shown in the image below:

![Image](https://video.udacity-data.com/topher/2018/May/5aefb2c0_diapositiva6-1/diapositiva6-1.png)

Finally, if we follow the section of the image that corresponds to the fourth window through this new CNN, we see the corresponding result in the last layer, as shown in the image below:

![Image](https://video.udacity-data.com/topher/2018/May/5aefb335_diapositiva8-1/diapositiva8-1.png)

In fact, if we follow all the windows through the CNN we see that all the 16 windows are contained within the last layer of this new CNN. Therefore, passing the 16 windows individually through the old CNN is exactly the same as passing the whole image only once through this new CNN.

![Image](https://video.udacity-data.com/topher/2018/May/5af331d6_last/last.png)

This is how you can apply sliding windows with a CNN. This technique makes the whole process much more efficient. However, this technique has a downside: the position of the bounding boxes is not going to be very accurate. The reason is that it is quite unlikely that a given size window and stride will be able to match the objects in the images perfectly. In order to increase the accuracy of the bounding boxes, YOLO uses a grid instead of sliding windows, in addition to two other techniques, known as Intersection Over Union and Non-Maximal Suppression.

The combination of the above techniques is part of the reason the YOLO algorithm works so well. Before diving into how YOLO puts all these techniques together, we will look first at each technique individually.

### Grid Method
[Detailed Explaination Video](https://www.youtube.com/watch?v=OmgR35Go79Y)

As sliding window is too slow, YOLO uses a grid to improve localization. For each grid cells, we have an associated vector that tells us if an object is in that cell, the predicted bounding box and the class of the object. 

#### Training on a Grid
[Detailed Explaination Video](https://www.youtube.com/watch?v=uhefpakvXh8)

For each training image, we break into each grid and assign a ground truth. We design a CNN with the `height of grids * width of grids * size of feature vector` and put it into the CNN to train.

#### Generating Bounding Boxes
[Detailed Explaination Video](https://www.youtube.com/watch?v=TGfPX-XcyOs)

For each training image, we locate the midpoint of each object in the image and assign the ground true bounding box to the grid cell that contains the mid point. In YOLO, (x, y) determine the distance the centre point is from the top left of the bounding box. While width and height are based on precentage compared to the image.
