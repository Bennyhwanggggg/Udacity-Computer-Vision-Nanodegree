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
