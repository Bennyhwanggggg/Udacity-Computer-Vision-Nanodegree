# Types of Features and Image Segmentation

## Corner Detectors
[Detailed explaination video](https://youtu.be/jemzDq07MEI)

Corners are detected using gradient measurements. Corners are intersection of two edges, so we can look at cell where gradient is high in all directions. This can be computed by Sobel operators.
A common corner detectors that depend on intensity is **Harris Corner Detector**.

A corner can be located by following these steps:

- Calculate the gradient for a small window of the image, using sobel-x and sobel-y operators (without applying binary thesholding).
- Use vector addition to calculate the magnitude and direction of the total gradient from these two values. ![Image](https://video.udacity-data.com/topher/2019/February/5c5b4a8b_vector-addition/vector-addition.png)
- Apply this calculation as you slide the window across the image, calculating the gradient of each window. When a big variation in the direction & magnitude of the gradient has been detected - a corner has been found!

## Dilation and Erosion
Dilation and erosion are known as **morphological operations**. They are often performed on binary images, similar to contour detection. Dilation enlarges bright, white areas in an image by adding pixels to the perceived boundaries of objects in that image. Erosion does the opposite: it removes pixels along object boundaries and shrinks the size of objects.

Often these two operations are performed in sequence to enhance important object traits!

### Dilation
To dilate an image in OpenCV, you can use the `dilate` function and three inputs: an original binary image, a kernel that determines the size of the dilation (None will result in a default size), and a number of iterations to perform the dilation (typically = 1). In the below example, we have a 5x5 kernel of ones, which move over an image, like a filter, and turn a pixel white if any of its surrounding pixels are white in a 5x5 window! We’ll use a simple image of the cursive letter “j” as an example.
```
# Reads in a binary image
image = cv2.imread(‘j.png’, 0) 

# Create a 5x5 kernel of ones
kernel = np.ones((5,5),np.uint8)

# Dilate the image
dilation = cv2.dilate(image, kernel, iterations = 1)
```

### Erosion
To erode an image, we do the same but with the `erode` function.
```
# Erode the image
erosion = cv2.erode(image, kernel, iterations = 1)
```
![Image](https://video.udacity-data.com/topher/2017/June/5956cf7e_screen-shot-2017-06-30-at-3.22.40-pm/screen-shot-2017-06-30-at-3.22.40-pm.png)

The letter "j": (left) erosion, (middle) original image, (right) dilation

### Opening
As mentioned, above, these operations are often combined for desired results! One such combination is called **opening**, which is **erosion followed by dilation**. This is useful in noise reduction in which erosion first gets rid of noise (and shrinks the object) then dilation enlarges the object again, but the noise will have disappeared from the previous erosion!

To implement this in OpenCV, we use the function `morphologyEx` with our original image, the operation we want to perform, and our kernel passed in.
```
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
```
![Image](https://video.udacity-data.com/topher/2017/June/5956df32_screen-shot-2017-06-30-at-4.30.11-pm/screen-shot-2017-06-30-at-4.30.11-pm.png)

### Closing
**Closing** is the reverse combination of opening; it’s **dilation followed by erosion**, which is useful in closing small holes or dark areas within an object.

Closing is reverse of Opening, Dilation followed by Erosion. It is useful in closing small holes inside the foreground objects, or small black points on the object.
```
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
```
![Image](https://video.udacity-data.com/topher/2017/June/5956e0b9_screen-shot-2017-06-30-at-4.37.13-pm/screen-shot-2017-06-30-at-4.37.13-pm.png)


Many of these operations try to extract better (less noisy) information about the shape of an object or enlarge important features, as in the case of corner detection!

## Image Segmentation
Grouping or segmenting images into distinct parts is known as image segmentation.

The simplest case for image segmentation is in background subtraction. In video and other applications, it is often the case that a human has to be isolated from a static or moving background, and so we have to use segmentation methods to distinguish these areas. Image segmentation is also used in a variety of complex recognition tasks, such as in classifying every pixel in an image of the road.

We'll look at a couple ways to segment an image:
- Using contours to draw boundaries around different parts of an image
- Clustering image data by some measure of color or texture similarity.

![Image](https://video.udacity-data.com/topher/2018/April/5ad9254f_screen-shot-2018-04-19-at-4.24.35-pm/screen-shot-2018-04-19-at-4.24.35-pm.png)

Partially-segmented image of a road; the image separates areas that contain a pedestrian from areas in the image that contain the street or cars.

### Image Contours
[Detailed Explaination Video](https://www.youtube.com/watch?time_continue=4&v=Wcbrl7Wr_kU&feature=emb_logo)

One common technique is [K-Means Clustering](https://www.youtube.com/watch?v=Cf_LSDCEBzk). It is an unsupervised learning technique and can be used to separate an image into segments by clustering data points that have similar traits.
