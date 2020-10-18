# Robot Localization

## Probabilities
### Quantifying Certainty and Uncertainty
When we talk about being certain that a robot is at a certain location (x, y), moving a certain direction, or sensing a certain environment, we can quantify that certainty using probabilistic quantities. Sensor measurements and movement all have some uncertainty associated with them (ex. a speedometer that reads 50mph may be off by a few mph, depending on whether a car is moving up or down hill).

![Image](https://video.udacity-data.com/topher/2018/May/5aea99c9_screen-shot-2018-05-02-at-10.10.16-pm/screen-shot-2018-05-02-at-10.10.16-pm.png)

### Bayes' Rule
Bayes' Rule is extremely important in robotics and it can be summarized in one sentence: given an initial prediction, if we gather additional data (data that our initial prediction depends on), we can improve that prediction!

#### Initial Scenario

![Image](https://video.udacity-data.com/topher/2018/May/5aea9874_screen-shot-2018-05-02-at-10.04.31-pm/screen-shot-2018-05-02-at-10.04.31-pm.png)

Map of the road and the initial location prediction.

We know a little bit about the map of the road that a car is on (pictured above). We also have an initial GPS measurement; the GPS signal says the car is at the red dot. However, this GPS measurement is inaccurate up to about 5 meters. So, the vehicle could be located anywhere within a 5m radius circle around the dot.

#### Sensors
Then we gather data from the car's sensors. Self-driving cars mainly use three types of sensors to observe the world:

- Camera, which records video,
- Lidar, which is a light-based sensor, and
- Radar, which uses radio waves.

All of these sensors detect surrounding objects and scenery.

Autonomous cars also have lots of **internal sensors** that measure things like the speed and direction of the car's movement, the orientation of its wheels, and even the internal temperature of the car!

##### Sensor Measurements
Suppose that our sensors detect some details about the terrain and the way our car is moving, specifically:

- The car could be anywhere within the GPS 5m radius circle,
- The car is moving upwards on this road,
- There is a tree to the left of our car, and
- The car’s wheels are pointing to the right.

Knowing only these sensor measurements, examine the map below and answer the following quiz question.

![Image](https://video.udacity-data.com/topher/2018/May/5aea98f8_screen-shot-2018-05-02-at-10.06.44-pm/screen-shot-2018-05-02-at-10.06.44-pm.png)

Road map with additional sensor data

### Reducing Uncertainty
[Detailed Explaination Video](https://www.youtube.com/watch?v=vhl-SADfti8)

We can use sensor data to improve our estimates of car location using Baye's rule. 

### What is a Probability Distribution?
Probability distributions allow you to represent the probability of an event using a mathematical equation. Like any mathematical equation:

- probability distributions can be **visualized** using a graph especially in 2-dimensional cases.
- probability distributions can be **worked with using algebra, linear algebra and calculus**.

These distributions make it much easier to understand and summarize the probability of a system whether that system be a coin flip experiment or the location of an autonomous vehicle.

### Types of Probability Distributions
Probability distributions are really helpful for understanding the probability of a system. Looking at the big pictures, there are two types of probability distributions:

- discrete probability distributions
- continuous probability distributions
Before we get into the details about what discrete and continuous mean, take a look at these two visualizations below. The first image shows a discrete probability distribution and the second a continuous probability distribution. What is similar and what is different about each visualization?

![Image](https://video.udacity-data.com/topher/2018/May/5aeb5145_screen-shot-2018-05-03-at-11.13.10-am/screen-shot-2018-05-03-at-11.13.10-am.png)

Discrete Distribution (left) and Continuous Distribution (right).

## Localization 
[Detailed Explaination Video Part 1](https://www.youtube.com/watch?v=OB6GZxfvESw)

[Detailed Explaination Video Part 2](https://www.youtube.com/watch?v=RCEieE2t8U4)

[Detailed Explaination Video Part 3](https://www.youtube.com/watch?v=aWQMJQQmNGw)

[Detailed Explaination Video Part 4](https://www.youtube.com/watch?v=ZGvmFn_u56o)

[Detailed Explaination Video Part 5](https://www.youtube.com/watch?v=eIjyrQpDogg)

[Detailed Explaination Video Part 6](https://www.youtube.com/watch?v=UX3W8TUKbJ0)

[Detailed Explaination Video Part 7](https://www.youtube.com/watch?v=GqWszyHTYas)

[Detailed Explaination Video Part 8](https://www.youtube.com/watch?v=UX3W8TUKbJ0)

[Detailed Explaination Video Part 9](https://www.youtube.com/watch?v=gDO4sF8gR9k)

[Detailed Explaination Video Part 10](https://www.youtube.com/watch?v=-3qTapGGa-8)

## Robot Motion
[Detailed Explaination Video](https://www.youtube.com/watch?v=mNXm1wjTumY)

### Move Function
[Detailed Explaination Video Part 1](https://www.youtube.com/watch?v=wfjE0mVADIk)

[Detailed Explaination Video Part 2](https://www.youtube.com/watch?v=TnFq6hufsYs)

### Inexact Motion
[Detailed Explaination Video Part 1](https://www.youtube.com/watch?v=hHAwFNsIp1c)

[Detailed Explaination Video Part 2](https://www.youtube.com/watch?v=68Kao9dkIKA)

[Detailed Explaination Video Part 3](https://www.youtube.com/watch?v=QCnPJcNprEU)

### Limit Distributions
[Detailed Explaination Video](https://www.youtube.com/watch?v=NJvalJwjz18)

### Sense and Move
[Detailed Explaination Video Part 1](https://www.youtube.com/watch?v=v2dYzm6-YVs)
[Detailed Explaination Video Part 2](https://www.youtube.com/watch?v=1s2dRczcu1A)

### Localization Summary
![Image](https://video.udacity-data.com/topher/2018/May/5b073c02_sense-move/sense-move.png)

#### Sense/Move Cycle
1. When a robot senses, a **measurement update** happens; this is a simple multiplication that is based off of Bayes' rule, which says that we can update our belief based on measurements! This step was also followed by a **normalization** that made sure the resultant distribution was still vald (and added up to 1).
2. When it moves, a **motion update** or prediction step occurs; this step is a convolution that shifts the distribution in the direction of motion.

After this cycle, we are left with an altered **posterior** distribution!

![Image](https://video.udacity-data.com/topher/2018/May/5b073c5a_prob-dists/prob-dists.png)

A move sense cycle in action, with an initial belief at the top.


