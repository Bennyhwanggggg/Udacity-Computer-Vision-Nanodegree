# Types of Features and Image Segmentation

## Corner Detectors
[Detailed explaination video](https://youtu.be/jemzDq07MEI)

Corners are detected using gradient measurements. Corners are intersection of two edges, so we can look at cell where gradient is high in all directions. This can be computed by Sobel operators.
A common corner detectors that depend on intensity is **Harris Corner Detector**.

A corner can be located by following these steps:

- Calculate the gradient for a small window of the image, using sobel-x and sobel-y operators (without applying binary thesholding).
- Use vector addition to calculate the magnitude and direction of the total gradient from these two values. ![Image](https://video.udacity-data.com/topher/2019/February/5c5b4a8b_vector-addition/vector-addition.png)
- Apply this calculation as you slide the window across the image, calculating the gradient of each window. When a big variation in the direction & magnitude of the gradient has been detected - a corner has been found!
