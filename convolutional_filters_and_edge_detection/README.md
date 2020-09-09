# Convolutional Filters and Edge Detection
## Filters
In addition to taking advantage of color information, we can make use of patterns of grayscale intensity in an image. Intensity is a measure of light and dark similar to brightness, and we can use this knowledge to detect other areas or objects of interest. For example, you can often identify the edges of an object by looking at an abrupt change in intensity, which happens when an image changes from a very dark to light area, or vice versa.

To detect these changes, you’ll be using and creating specific image filters that look at groups of pixels and detect big changes in intensity in an image. These filters produce an output that shows these edges.

So, let’s take a closer look at these filters and see when they’re useful in processing images and identifying traits of interest.

## Frequency in images
We have an intuition of what frequency means when it comes to sound. High-frequency is a high pitched noise, like a bird chirp or violin. And low frequency sounds are low pitch, like a deep voice or a bass drum. For sound, frequency actually refers to how fast a sound wave is oscillating; oscillations are usually measured in cycles/s (Hz), and high pitches and made by high-frequency waves. Examples of low and high-frequency sound waves are pictured below. On the y-axis is amplitude, which is a measure of sound pressure that corresponds to the perceived loudness of a sound and on the x-axis is time.

![Image](https://video.udacity-data.com/topher/2018/April/5ad8ff83_screen-shot-2018-04-19-at-1.43.30-pm/screen-shot-2018-04-19-at-1.43.30-pm.png)

(Top image) a low frequency sound wave (bottom) a high frequency sound wave.

## High and low frequency
Similarly, frequency in images is a rate of change. But, what does it means for an image to change? Well, images change in space, and a high frequency image is one where the intensity changes a lot. And the level of brightness changes quickly from one pixel to the next. A low frequency image may be one that is relatively uniform in brightness or changes very slowly. This is easiest to see in an example.

![Image](https://video.udacity-data.com/topher/2018/April/5ad8ffce_screen-shot-2018-04-19-at-1.44.37-pm/screen-shot-2018-04-19-at-1.44.37-pm.png)

High and low frequency image patterns.

Most images have both high-frequency and low-frequency components. In the image above, on the scarf and striped shirt, we have a high-frequency image pattern; this part changes very rapidly from one brightness to another. Higher up in this same image, we see parts of the sky and background that change very gradually, which is considered a smooth, low-frequency pattern.

**High-frequency components also correspond to the edges of objects in images**, which can help us classify those objects.

## Fourier Transform
The Fourier Transform (FT) is an important image processing tool which is used to decompose an image into its frequency components. The output of an FT represents the image in the frequency domain, while the input image is the spatial domain (x, y) equivalent. In the frequency domain image, each point represents a particular frequency contained in the spatial domain image. So, for images with a lot of high-frequency components (edges, corners, and stripes), there will be a number of points in the frequency domain at high frequency values.

Take a look at how FT's are done with OpenCV, [here](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html).

![Image](https://video.udacity-data.com/topher/2018/April/5adee70e_screen-shot-2018-04-24-at-1.12.46-am/screen-shot-2018-04-24-at-1.12.46-am.png)

An image of a soccer player and the corresponding frequency domain image (right). The concentrated points in the center of the frequency domain image mean that this image has a lot of low frequency (smooth background) components.

This decomposition is particularly interesting in the context of bandpass filters, which can isolate a certain range of frequencies and mask an image according to a low and high frequency threshold.

## Edge Handling
Kernel convolution relies on centering a pixel and looking at it's surrounding neighbors. So, what do you do if there are no surrounding pixels like on an image corner or edge? Well, there are a number of ways to process the edges, which are listed below. It’s most common to use padding, cropping, or extension. In extension, the border pixels of an image are copied and extended far enough to result in a filtered image of the same size as the original image.

**Extend** The nearest border pixels are conceptually extended as far as necessary to provide values for the convolution. Corner pixels are extended in 90° wedges. Other edge pixels are extended in lines.

**Padding** The image is padded with a border of 0's, black pixels.

**Crop** Any pixel in the output image which would require values from beyond the edge is skipped. This method can result in the output image being slightly smaller, with the edges having been cropped.

## Gradients
Gradients are a measure of intensity change in an image, and they generally mark object boundaries and changing area of light and dark. If we think back to treating images as functions, F(x, y), we can think of the gradient as a derivative operation F’(x, y). Where the derivative is a measurement of intensity change.

### Sobel filters
The Sobel filter is very commonly used in edge detection and in finding patterns in intensity in an image. Applying a Sobel filter to an image is a way of taking (an approximation) of the derivative of the image in the xx or yy direction. The operators for Sobel_x and Sobel_y, respectively, look like this:

![Image](https://video.udacity-data.com/topher/2017/June/59518439_screen-shot-2017-06-26-at-2.35.11-pm/screen-shot-2017-06-26-at-2.35.11-pm.png)


Sobel filters

Next, let's see an example of these two filters applied to an image of the brain.

![Image](https://video.udacity-data.com/topher/2017/June/595186d6_screen-shot-2017-06-26-at-3.11.58-pm/screen-shot-2017-06-26-at-3.11.58-pm.png)

Sobel x and y filters (left and right) applied to an image of a brain

**x vs. y**
In the above images, you can see that the gradients taken in both the xx and the yy directions detect the edges of the brain and pick up other edges. Taking the gradient in the xx direction emphasizes edges closer to vertical. Alternatively, taking the gradient in the yy direction emphasizes edges closer to horizontal.

### Magnitude
Sobel also detects which edges are strongest. This is encapsulated by the magnitude of the gradient; the greater the magnitude, the stronger the edge is. The magnitude, or absolute value, of the gradient is just the square root of the squares of the individual x and y gradients. For a gradient in both the xx and yy directions, the magnitude is the square root of the sum of the squares.

abs_sobelx= \sqrt{(sobel_x)^2}

abs_sobely= \sqrt{(sobel_y)^2}

abs_sobelxy= \sqrt{(sobel_x)^2+(sobel_y)^2}

### Direction
In many cases, it will be useful to look for edges in a particular orientation. For example, we may want to find lines that only angle upwards or point left. By calculating the direction of the image gradient in the x and y directions separately, we can determine the direction of that gradient!

The direction of the gradient is simply the inverse tangent (arctangent) of the yy gradient divided by the xx gradient:

tan^{-1}{(sobel_y/sobel_x)}

### Low Pass Filter - Gaussian Blur
We often want to use a low pass filter to remove noise first before using a high pass filter to detect edges. Low pass filters smooth/blurs out the image by averaging the pixels so there is not as high change in pixel values due to noise. Gaussian Blur is an example of low pass filter.

### The Importance of Filters
What you've just learned about different types of filters will be really important as you progress through this course, especially when you get to Convolutional Neural Networks (CNNs). CNNs are a kind of deep learning model that can learn to do things like image classification and object recognition. They keep track of spatial information and learn to extract features like the edges of objects in something called a convolutional layer. Below you'll see an simple CNN structure, made of multiple layers, below, including this "convolutional layer".

![Image](https://video.udacity-data.com/topher/2018/May/5b1070e4_screen-shot-2018-05-31-at-2.59.36-pm/screen-shot-2018-05-31-at-2.59.36-pm.png)

Layers in a CNN.

Convolutional Layer
The convolutional layer is produced by applying a series of many different image filters, also known as convolutional kernels, to an input image.

![Image](https://video.udacity-data.com/topher/2018/May/5b10723a_screen-shot-2018-05-31-at-3.06.07-pm/screen-shot-2018-05-31-at-3.06.07-pm.png)

4 kernels = 4 filtered images.

In the example shown, 4 different filters produce 4 differently filtered output images. When we stack these images, we form a complete convolutional layer with a depth of 4!

![Image](https://video.udacity-data.com/topher/2018/May/5b10729b_screen-shot-2018-05-31-at-3.07.03-pm/screen-shot-2018-05-31-at-3.07.03-pm.png)

A convolutional layer.

## Edge Detection
Now that you've seen how to define and use image filters for smoothing images and detecting the edges (high-frequency) components of objects in an image, let's move one step further. The next few videos will be all about how we can use what we know about pattern recognition in images to begin identifying unique shapes and then objects.

### Edges to Boundaries and Shapes
We know how to detect the edges of objects in images, but how can we begin to find unifying boundaries around objects? We'll want to be able to do this to separate and locate multiple objects in a given image. Next, we'll discuss the Hough transform, which transforms image data from the x-y coordinate system into Hough space, where you can easily identify simple boundaries like lines and circles.

The Hough transform is used in a variety of shape-recognition applications, as seen in the images pictured below. On the left you see how a Hough transform can find the edges of a phone screen and on the right you see how it's applied to an aerial image of farms (green circles in this image).

![Image](https://video.udacity-data.com/topher/2018/April/5ad90d4d_screen-shot-2018-04-19-at-2.42.19-pm/screen-shot-2018-04-19-at-2.42.19-pm.png)

Hough transform applied to phone-edge and circular farm recognition.
