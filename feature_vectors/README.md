# Feature Vectors

## Feature Vectors, Corner and Object Detections
[Detailed Explaination Video](https://www.youtube.com/watch?v=-PF1_MITrOw)

Now, we've seen examples of shape-based features, like corners, that can be extracted from images, but how can we actually uses the features to detect whole objects?

Well, let's think about an example we've seen of corner detection for an image of a mountain.

![Image](https://video.udacity-data.com/topher/2018/April/5ad92b7d_screen-shot-2018-04-19-at-4.47.30-pm/screen-shot-2018-04-19-at-4.47.30-pm.png)

Corner detection on an image of Mt. Rainier

Say we want a way to detect this mountain in other images, too. A single corner will not be enough to identify this mountain in any other images, but, we can take a set of features that define the shape of this mountain, group them together into an array or vector, and then use that set of features to create a mountain detector!

## Real Time Feature Detection
[Detailed Explaination Video](https://www.youtube.com/watch?v=zPxylrXf-Gs)

### ORB (Oriented Fast and Rotated Brief)
[Detailed Explaination Video](https://www.youtube.com/watch?v=WN37zcMhMas)

ORB starts by trying to find distinctive small regions called keypoints in the image. Once image keypoints are calculated, ORB computes binary feature vectors for them, which represents intensity around them.

#### FAST (Features from Accelerated Segment Test)
[Detailed Explaination Video](https://www.youtube.com/watch?v=DCHAc6fjcVM)

Used to find keypoints in the ORB algorithm. FAST compares the region of intensity around a pixel and uses a threshold to compute the intensity difference. A pixel is a keypoint if more than 8 connected pixel in a circle around the pixel are darker or brighter than the target pixel. 
The keypoints found by FAST give us the information about key edges in an image that defines an object.

#### BRIEF (Binary Robust Independent Elementary Features)
[Detailed Explaination Video](https://www.youtube.com/watch?v=EKIPEPpRciw)

Uses Binary feature vectors to describe the image. Starts by smoothing the given image with a Gaussian kernel so the descriptor is not too sensitive to high frequency noise. Next, given a keypoint, BRIEF selects a neighbourhood patch around the keypoint. In that patch, it selects random pixels centered around the keypoint and construct the binary vector based on comparison between the two keypoints. It does this for 256 points for each keypoint until the 256 length binary vector is constructed.

#### Scale and Rotation Invariance
[Detailed Explaination Video](https://www.youtube.com/watch?v=2k3T6rfjvx0)

Given an image, the ORB algorithm, starts building a pyramid of the original consists the scaled downed version of the original image by resolution. At each level, the resolution is halfed. At each level, ORB runs FAST to get keypoints in each level, such that ORB is getting keypoints at different scale. This is important since object can be at any distance away from camera.

ORB also calculate rotation using orientation centroids. Once orientation has been assigned to each keypoint at original level, orb does the same for each level of the pyramid. This result in each keypoint been different size. 

After this, ORB uses **rotated BRIEF** to make the rotation rotation invariant. RBRIEF then construct set of feature vectors known as ORB descriptor. 

#### Feature Matching
[Detailed Explaination Video](https://www.youtube.com/watch?v=RH05Wnl1-2A)

Given a training image and a query image, the ORB descripor is constructed for the training image and query image. Then we perform keypoints matching which is typically performed by a matching function. Matching function rates the quality of the match by some metric, such as Euclidian distance, to determine a match. Hemming metric is typically used for ORB. It counts the number of matching bits to determine if there's a match.
