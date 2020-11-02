# State and Motion

## Localization
All self-driving cars go through the same series of steps to safely navigate through the world.
You’ve been working on the first step: **localization**. Before cars can safely navigate, they first use sensors and other collected data to best estimate where they are in the world.

## Kalman Filter
Let’s review the steps that a Kalman filter takes to localize a car.

### 1. Initial Prediction
First, we start with an initial prediction of our car’s location and a probability distribution that describes our uncertainty about that prediction.

Below is a 1D example, we know that our car is on this one lane road, but we don't know its exact location.

![Image](https://video.udacity-data.com/topher/2017/September/59a9f384_screen-shot-2017-09-01-at-4.55.27-pm/screen-shot-2017-09-01-at-4.55.27-pm.png)

A one lane one and an initial, uniform probability distribution.

### 2. Measurement Update
We then sense the world around the car. This is called the measurement update step, in which we gather more information about the car’s surroundings and refine our location prediction.

Say, we measure that we are about two grid cells in front of the stop sign; our measurement isn't perfect, but we have a much better idea of our car's location.

![Image](https://video.udacity-data.com/topher/2017/September/59a9f4a7_screen-shot-2017-09-01-at-4.59.24-pm/screen-shot-2017-09-01-at-4.59.24-pm.png)

Measurement update step.

### 3. Prediction (or Time Update)
The next step is moving. Also called the time update or prediction step; we predict where the car will move, based on the knowledge we have about its velocity and current position. And we shift our probability distribution to reflect this movement.

In the next example, we shift our probability distribution to reflect a one cell movement to the right.

![Image](https://video.udacity-data.com/topher/2017/September/59a9f57b_screen-shot-2017-09-01-at-5.03.57-pm/screen-shot-2017-09-01-at-5.03.57-pm.png)

Prediction step.

### 4. Repeat
Then, finally, we’ve formed a new estimate for the position of the car! The Kalman Filter simply repeats the sense and move (measurement and prediction) steps to localize the car as it’s moving!

![Image](https://video.udacity-data.com/topher/2017/September/59a9fa11_screen-shot-2017-09-01-at-5.23.18-pm/screen-shot-2017-09-01-at-5.23.18-pm.png)

Kalman Filter steps.

### The Takeaway
The beauty of Kalman filters is that they combine somewhat inaccurate sensor measurements with somewhat inaccurate predictions of motion to get a filtered location estimate **that is better than any estimates that come from only sensor readings or only knowledge about movement**.

## What is State?
When you localize a car, you’re interested in only the car’s position and it’s movement.

This is often called the state of the car.
- The state of any system is a set of values that we care about.

In the cases we’ve been working with, the state of the car includes the car’s current **position, x**, and its **velocity, v**.

In code this looks something like this:
```
x = 4
vel = 1

state = [x, vel]
```
### Predicting Future States
The state gives us most of the information we need to form predictions about a car’s future location. And in this lesson, we’ll see how to represent state and how it changes over time.

For example, say that our world is a one-lane road, and we know that the current position of our car is at the start of this road, at the 0m mark. We also know the car’s velocity: it’s moving forward at a rate of 50m/s. These values are it’s initial state.
```
state = [0, 50]
```
![Image](https://video.udacity-data.com/topher/2017/September/59aa0a10_screen-shot-2017-09-01-at-6.31.45-pm/screen-shot-2017-09-01-at-6.31.45-pm.png)
The estimate of the initial state of the car.
