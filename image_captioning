# Image Captioning
[Introduction](https://www.youtube.com/watch?v=dobNslC2y-o)

## COCO Dataset
The COCO dataset is one of the largest, publicly available image datasets and it is meant to represent realistic scenes. What I mean by this is that COCO does not overly pre-process images, instead these images come in a variety of shapes with a variety of objects and environment/lighting conditions that closely represent what you might get if you compiled images from many different cameras around the world.

To explore the dataset, you can check out the [dataset website](https://cocodataset.org/#explore).

### Explore
Click on the explore tab and you should see a search bar that looks like the image below. Try selecting an object by it's icon and clicking search!

![Image](https://video.udacity-data.com/topher/2018/May/5aeb88bb_screen-shot-2018-05-03-at-3.09.44-pm/screen-shot-2018-05-03-at-3.09.44-pm.png)

A sandwich is selected by icon.

You can select or deselect multiple objects by clicking on their corresponding icon. Below are some examples for what a `sandwich` search turned up! You can see that the initial results show colored overlays over objects like sandwiches and people and the objects come in different sizes and orientations.

![Image](https://video.udacity-data.com/topher/2018/May/5aeb8986_screen-shot-2018-05-03-at-3.12.38-pm/screen-shot-2018-05-03-at-3.12.38-pm.png)

COCO sandwich detections

### Captions
COCO is a richly labeled dataset; it comes with class labels, labels for segments of an image, and a set of captions for a given image. To see the captions for an image, select the text icon that is above the image in a toolbar. Click on the other options and see what the result is.

![Image](https://video.udacity-data.com/topher/2018/May/5aeb8a8e_screen-shot-2018-05-03-at-3.15.20-pm/screen-shot-2018-05-03-at-3.15.20-pm.png)

Example captions for an image of people at a sandwich counter.

When we actually train our model to generate captions, we'll be using these images as input and sampling one caption from a set of captions for each image to train on.

## CNN-RNN Model
[Detailed Explaination Video](https://www.youtube.com/watch?v=n7kdMiX1Xz8)

A training image and an associated caption are used as an input. At end of the CNN model, instead of classifying the image, we remove the final classifying image and pass it into RNN. The CNN acts like the encoder to produce a feature vector as the input for the RNN.

### The Glue, Feature Vector
[Detailed Explaination Video](https://www.youtube.com/watch?v=u2ZdcUDnHm0)

Sometimes we can put the feature vector into an untrained linear layer before passing it in as the input into RNN. The RNN will decode the processed feature vector to generate the caption.

### RNN
#### Tokenizing Captions
[Detailed Explaination Video](https://www.youtube.com/watch?v=aeEFb0eSzJ8)

Since Nerual Network does not do well with strings, we need to tokenize words into a list of integers by mapping each words into some numerical index. 

#### Tokenizing Words
##### Words to Vectors
At this point, we know that you cannot directly feed words into an LSTM and expect it to be able to train or produce the correct output. These words first must be turned into a numerical representation so that a network can use normal loss functions and optimizers to calculate how "close" a predicted word and ground truth word (from a known, training caption) are. So, we typically turn a sequence of words into a sequence of numerical values; a vector of numbers where each number maps to a specific word in our vocabulary.

To process words and create a vocabulary, we'll be using the Python text processing toolkit: NLTK. in the below video, we have one of our content developers, Arpan, explain the concept of word tokenization with NLTK.

[NLTK Video](https://www.youtube.com/watch?v=4Ieotbeh4u8)

#### RNN Training
[Detaild Explaination Video](https://www.youtube.com/watch?v=P-tHxD7kRmA)

##### Training vs. Testing
During training, we have a true caption which is fixed, but during testing the caption is being actively generated (starting with `<start>`), and at each step you are getting the most likely next word and using that as input to the next LSTM cell.

###### Caption Generation, Test Data
After the CNN sees a new, test image, the decoder should first produce the `<start>` token, then an output distribution at each time step that indicates the most likely next word in the sentence. We can sample the output distribution (namely, extract the one word in the distribution with the highest probability of being the next word) to get the next word in the caption and keep this process going until we get to another special token: `<end>`, which indicates that we are done generating a sentence.

## Video Captioning
[Detailed Explaination Voideo](https://www.youtube.com/watch?v=I_m9JyKTfbQ)
In the case of video captioning, the only step that has to change is the feature vector which would be a series of image frame. We merge all the frame into a single vector by taking the average or using some other method and putting it into the RNN in the same way as before.
