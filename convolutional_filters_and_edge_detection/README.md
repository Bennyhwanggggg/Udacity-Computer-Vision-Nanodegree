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
