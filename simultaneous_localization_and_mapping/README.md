# Simultaneous Localization and Mapping

In the previous lessons, you learned all about localization methods that aim to locate a robot or car in an environment given sensor readings and motion data, and we would start out knowing close to nothing about the surrounding environment. In practice, in addition to localization, we also want to build up a model of the robot's environment so that we have an idea of objects, and landmarks that surround it and so that we can use this map data to ensure that we are on the right path as the robot moves through the world!

In this lesson, you'll learn how to implement SLAM (Simultaneous Localization and Mapping) for a 2 dimensional world! You’ll combine what you know about robot sensor measurements and movement to create locate a robot and create a map of an environment from only sensor and motion data gathered by a robot, over time. SLAM gives you a way to track the location of a robot in the world in real-time and identify the locations of landmarks such as buildings, trees, rocks, and other world features.

## Graph SLAM
[Read more here](https://medium.com/@krunalkshirsagar/graph-slam-a-noobs-guide-to-simultaneous-localization-and-mapping-aaff4ee91dee)

One of the SLAM method. It is a graph-based SLAM approach constructs a simplified estimation problem by abstracting the raw sensor measurements. These raw measurements are replaced by the edges in the graph which can then be seen as virtual measurements. But according to Kalman filters and various other robotic techniques, we have learned that the location is actually uncertain. So, rather than assuming in our X-Y coordinate system the robot moved to the right by 10 exactly, it’s better to understand that the actual location of the robot after the x1= x0+10 motion update is a Gaussian centered around (10,0), but it’s possible that the robot is somewhere else.

![Image](https://miro.medium.com/max/875/1*jJFMFqy51YYQUczkxF4Ylg.png)

![Image](https://miro.medium.com/max/875/1*fybZIa9-vdMM7naPXP54MA.png)

The product of these two Gaussian is now our constraint. The goal is to maximize the likelihood of the position x1 given the position x0 is (0,0). So, what Graph SLAM does is, it defines the probabilities using a sequence of such constraints. Say we have a robot that moves in some space, GRAPH SLAM collects its initial location which is (0,0) initially, also called as Initial Constraints, then collects lots of relative constraints that relate each robot pose to the previous robot pose also called as Relative Motion Constraints. As an example, let’s use landmarks that can be seen by the robot at various locations which would be Relative Measurement Constraints every time a robot sees a landmark. So, Graph SLAM collects those constraints in order to find the most likely configuration of the robot path along with the location of landmarks, and that is the mapping process.

![Image](https://miro.medium.com/max/875/1*7QMOYzsZu_IOkz4UK-7LTQ.png)
