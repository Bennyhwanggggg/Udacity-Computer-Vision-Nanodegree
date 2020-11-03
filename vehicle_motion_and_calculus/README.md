# Vehicle Motion and Calculus

## Navigation Sensors
We will be discussing the following sensors in this course:

- **Odometers** - An odometer measures how far a vehicle has traveled by counting wheel rotations. These are useful for measuring distance traveled (or displacement), but they are susceptible to bias (often caused by changing tire diameter). A "trip odometer" is an odometer that can be manually reset by a vehicle's operator.

- **Inertial Measurement Unit** - An Inertial Measurement Unit (or IMU) is used to measure a vehicle's heading, rotation rate, and linear acceleration using magnetometers, rate gyros, and accelerometers. We will discuss these sensors more in the next lesson.

## How Odometers Work
A mechanical odometer works by coupling the rotation of a vehicle's wheels to the rotation of numbers on a gauge like this:

![Image](https://video.udacity-data.com/topher/2017/December/5a3433fc_odometer/odometer.jpg)

Each of these numbers is written on a dial which has the numbers 0 - 9 written on it.

But that last digit on the gauge needs to rotate **very slowly** compared to the rotation rate of the vehicle's tires. Typically, a car's wheels will have to complete 750 rotations to move 1 mile. And since there are 10 digits on each dial, that means the last digit should only complete one rotation after the wheels have completed 7,500 rotations!

This reduction of rotation rate is accomplished through gear reduction. If you look at the blue and green gears in the image below you should get a sense for how that works.

## Position, Velocity, and Acceleration
Allow me to say the same thing about **position** and **velocity** in 5 different ways.

1. Velocity is the derivative of position.

2. Velocity is the instantaneous rate of change of position with respect to time.

3. An object's velocity tells us how much it's position will change when time changes.

4. Velocity at some time is just the slope of a line tangent to a graph of position vs. time

5. v(t) = dx/dt

It turns out you can say the same 5 things about **velocity** and **acceleration**.

1. Acceleration is the derivative of velocity.

2. Acceleration is the instantaneous rate of change of velocity with respect to time.

3. An object's acceleration tells us how much it's velocity will change when time changes.

4. Acceleration at some time is just the slope of a line tangent to a graph of velocity vs. time

5. a(t) = dv/dt

We can also make a couple interesting statements about the relationship between position and acceleration:

1. Acceleration is the second derivative of position.

We'll explore this more in the next lesson. For now, just know that differentiating position twice gives acceleration!
