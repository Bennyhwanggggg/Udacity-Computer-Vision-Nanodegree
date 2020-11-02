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

### Predicting State
Let’s look at the last example.

The initial state of the car is at the 0m position,and it's moving forward at a velocity of 50 m/s. Let’s assume that our car keeps moving forward at a constant rate.

Every second it moves 50m.

So, after three seconds, it will have reached the 150m mark and its velocity will not have changed (that's what a constant velocity means)!

![Image](https://video.udacity-data.com/topher/2017/September/59aa0a68_screen-shot-2017-09-01-at-6.33.03-pm/screen-shot-2017-09-01-at-6.33.03-pm.png)

Predicted state after 3 seconds have elapsed.

Its new, predicted state will be at the position **150m**, and with the velocity still equal to 50m/s.
```
predicted_state = [150, 50]
```

### Motion Model
This is a reasonable prediction, and we made it using:

1. The initial state of the car and
2. An assumption that the car is moving at a constant velocity.

This assumption is based on the physics equation:

```
distance_traveled = velocity * time
```

This equation is also referred to as a **motion model**. And there are many ways to model motion!

This motion model assumes constant velocity.

In our example, we were moving at a constant velocity of 50m/s for three seconds.

And we formed our new position estimate with the distance equation: `150m = 50m/sec*3sec`.

### The Takeaway
In order to predict where a car will be at a future point in time, you rely on a motion model.

### Uncertainty
It’s important to note, that no motion model is perfect; it’s a challenge to account for outside factors like wind or elevation, or even things like tire slippage, and so on.

But these models are still very important for localization.

Next, you’ll be asked to write a function that uses a motion model to predict a new state!

### More Complex Motion
Now, what if I gave you a more complex motion example?

And I told you that our car starts at the same point, at the 0m mark, and it’s moving 50m/s forward, but it’s also slowing down at a rate of 20m/s^2. This means it’s acceleration = -20m/s^2.

![Image](https://video.udacity-data.com/topher/2017/September/59aa0ae5_screen-shot-2017-09-01-at-6.35.20-pm/screen-shot-2017-09-01-at-6.35.20-pm.png)

Car moving at 50m/s and slowing down over time.

### Acceleration
So, if the car has a -20 m/s^2 acceleration, this means that:

If the car starts at a speed of 50m/s
At the next second, it will be going 50-20 or 30m/s and,
At the next second it will be going 30-20 or 10m/s.
This slowing down is also **continuous**, which means it happens gradually over time.

## Kinematics
Kinematics is the study of the motion of objects. Motion models are also referred to as kinematic equations, and these equations give you all the information you need to be able to predict the motion of a car.

Let's derive some of the most common motion models!

Constant Velocity
The constant velocity model assumes that a car moves at a constant speed. This is the simplest model for car movement.

**Example**

Say that our car is moving 100m/s, and we want to figure out how much it has moved from one point in time, t1, to another, t2. This is represented by the graph below.

![Image](https://video.udacity-data.com/topher/2017/September/59ac4407_screen-shot-2017-09-03-at-11.03.31-am/screen-shot-2017-09-03-at-11.03.31-am.png)

(Left) Graph of car velocity, (Right) a car going 100m/s on a road

### Displacement
How much the car has moved is called the **displacement** and we already know how to calculate this!

We know, for example, that if the difference between t2 and t1 is one second, then we'll have moved `100m/sec*1sec = 100m`. If the difference between t2 and t1 is two seconds, then we'll have moved `100m/sec*2sec = 200m`.

The displacement is always = `100m/sec*(t2-t1)`.

### Motion Model

Generally, for constant velocity, the motion model for displacement is:
```
displacement = velocity*dt
```
Where `dt` is calculus notation for "difference in time."

### Area Under the Line
Going back to our graph, displacement can also be thought of as the area under the line and within the given time interval.

![Image](https://video.udacity-data.com/topher/2017/September/59ac4cb8_screen-shot-2017-09-03-at-11.40.37-am/screen-shot-2017-09-03-at-11.40.37-am.png)

The area under the line, A, is equal to the displacement!

So, in addition to our motion model, we can also say that the displacement is equal to the area under the line!
```
displacement = A
```
### Constant Acceleration
The constant acceleration model is a little different; it assumes that our car is constantly accelerating; its velocity is changing at a constant rate.

Let's say our car has a velocity of 100m/s at time t1 and is accelerating at a rate of 10m/s^2.

![Image](https://video.udacity-data.com/topher/2017/September/59ac786a_screen-shot-2017-09-03-at-2.47.09-pm/screen-shot-2017-09-03-at-2.47.09-pm.png)

### Changing Velocity
For this motion model, we know that the velocity is constantly changing, and increasing +10m/s each second. This can be represented by this kinematic equation (where dv is the change in velocity):
```
dv = acceleration*dt
```
At any given time, this can also be written as the current velocity is the initial velocity + the change in velocity over some time (dv):
```
v = initial_velocity + acceleration*dt
```
### Displacement
Displacement can be calculated by finding the area under the line in between t1 and t2, similar to our constant velocity equation but a slightly different shape.

![Image](https://video.udacity-data.com/topher/2017/September/59ac79e6_screen-shot-2017-09-03-at-2.53.30-pm/screen-shot-2017-09-03-at-2.53.30-pm.png)

Area under the line, A1 and A2

This area can be calculated by breaking this area into two distinct shapes; a simple rectangle, A1, and a triangle, A2.

A1 is the same area as in the constant velocity model.
```
A1 = initial_velocity*dt
```
In other words, `A1 = 100m/s*(t2-t1)`.

A2 is a little trickier to calculate, but remember that the area of a triangle is `0.5*width*height`.

The width, we know, is our change in time (t2-t1) or `dt`.

And the height is the change in velocity over that time! From our earlier equation for velocity, we know that this value, dv, is equal to: `acceleration*(t2-t1)` or `acceleration*dt`

Now that we have the width and height of the triangle, we can calculate A2. Note that`**` is a Python operator for an exponent, so `**`2 is equivalent to `^2` in mathematics or squaring a value.
```
A2 = 0.5*acceleration*dt**2
```
### Motion Model
This means that our total displacement, A1+A2 ,can be represented by the equation:
```
displacement = initial_velocity*dt + 0.5*acceleration*dt**2
```
We also know that our velocity over time changes according to the equation:
```
dv = acceleration*dt
```
And these two equations, together, make up our motion model for constant acceleration.

## Different Motion Models
### Constant Velocity
In the first movement example, you saw that if we assumed our car was moving at a constant speed, 50 m/s, we came up with one prediction for it’s new state: at the 150 m mark, with no change in velocity.
```
# Constant velocity case

# initial variables
x = 0
velocity = 50
initial_state = [x, velocity]

# predicted state (after three seconds)
# this state has a new value for x, but the same velocity as in the initial state
dt = 3
new_x = x + velocity*dt
predicted_state = [new_x, velocity]  # predicted_state = [150, 50]
```
For this constant velocity model, we had:

- initial state = `[0, 50]`
- predicted state (after 3 seconds) = `[150, 50]`

### Constant Acceleration
But in the second case, we said that the car was slowing down at a rate of 20 m/s^2 and, after 3 seconds had elapsed, we ended up with a different estimate for its state.

To solve this localization problem, we had to use a different motion model and **we had to include a new value in our state: the acceleration of the car**.

The motion model was for constant acceleration:

- `distance = velocity*dt + 0.5*acceleration*dt^21 and
- `velocity = acceleration*dt`

The state includes acceleration in this model and looks like this: `[x, velocity, acc]`.
```
# Constant acceleration, changing velocity

# initial variables
x = 0
velocity = 50
acc = -20

initial_state = [x, velocity, acc]

# predicted state after three seconds have elapsed
# this state has a new value for x, and a new value for velocity (but the acceleration stays the same)
dt = 3

new_x = x + velocity*dt + 0.5*acc*dt**2
new_vel = velocity + acc*dt

predicted_state = [new_x, new_vel, acc]  # predicted_state = [60, -10, -20]
For this constant acceleration model, we had:

initial state = [0, 50, -20]
predicted state (after 3 seconds) = [60, -10, -20]
```

As you can see, our state and our state estimates vary based on the motion model we used and how we assumed the car was moving!

### How Many State Variables?
In fact, how many variables our state requires, depends on what motion model we are using.

For a constant velocity model, `x` and `velocity` will suffice.

But for a constant acceleration model, you'll also need our acceleration: `acc`.

But these are all just models.

### The Takeaway
For our state, we always choose the **smallest** representation (the smallest number of variables) that will work for our model.
