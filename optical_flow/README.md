# Optical Flow
[Detailed Explaination Video](https://www.youtube.com/watch?v=ciEo6PyMTeM)

Computer vision techniques to track an object over time. This includes the uncertainty of where the object may be located at a given time.

## Motion
[Detailed Explaination Video](https://www.youtube.com/watch?v=A-QJf04LBb0)

### Optical Flow
[Detailed Explaination Video](https://www.youtube.com/watch?v=TOS8UJwCtTg)

Optical flow is a method to track an object to help predict and classify mothions. It assumes pixels to not change signficantly between frames. It tracks a point to provide information about a movement and its speed.

#### Motion Vectors
[Detailed Explaination Video](https://www.youtube.com/watch?v=I3f3IEUI2tg)

A motion vector for a 2D image, has an x and y component (u, v). A motion vector for any point starts with the location of the point as the origin of the vector and it’s destination as the end of the vector (where the arrow point is).

All vectors have a direction and a magnitude. 

![Image](https://video.udacity-data.com/topher/2018/May/5b08663e_screen-shot-2018-05-25-at-12.38.14-pm/screen-shot-2018-05-25-at-12.38.14-pm.png)

A vector with `(u,v) = (-2, -6)`

#### Brightness Constancy Assumption
[Detailed Explaination Video](https://www.youtube.com/watch?v=GHz9Yzt5tro)

Optical flow assumes brightness don't change that much between frame which may not be true in practical applications. Therefore, we end up with an equation with intensity as a variable called the **Brightness Constancy Assumption**. 

You’ll note that the brightness constancy assumption gives us one equation with two unknowns (u and v), and so we also have to have another constraint; another equation or assumption that we can use to solve this problem.

Recall that in addition to assuming brightness constancy, we also assume that neighboring pixels have similar motion. Mathematically this means that pixels in a local patch have very similar motion vectors. For example, think of a moving person, if you choose to track a collection of points on that person’s face, all of those points should be moving at roughly the same speed. Your nose can’t be moving the opposite way of your chin.

This means that I shouldn’t get big changes in the flow vectors (u, v), and optical flow uses this idea of motion smoothness to estimate u and v for any point.

#### Tracking Features
[Detailed Explaination Video](https://www.youtube.com/watch?v=uFf6IZ5MxgU)

Optical flow will compute approximate motion vectors based on how the image intensity has changed over time. It will find matching feature points between frame using HOG or another edge detector. It then tries to locate where those key points can be found in different frame until it build up the path of the object over time.
