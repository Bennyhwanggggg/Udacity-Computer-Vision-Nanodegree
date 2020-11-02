# Kalman Filters

Kalman Filteris an algorithm that uses noisy sensor measurements (and Bayes' Rule) to produce reliable estimates of unknown quantities (like where a vehicle is likely to be in 3 seconds).

Kalman Filter is used to estimate **continuous state** while Monte Carlo Localization gives **discrete** state. As a result, it gives a uni-modal distribution while Monte Carlo Localization gives multi-modal distribution. 

![Image](https://video.udacity-data.com/topher/2018/May/5b09e381_screen-shot-2018-05-26-at-3.45.03-pm/screen-shot-2018-05-26-at-3.45.03-pm.png)

From the earlier positions, we can infer the velocity (indicated by the pink arrow in the image) of the object; the velocity appears to be u and to the left by a fairly consistent amount between each time step.

Assuming no drastic change in velocity occurs, we predict that at time t = 4, the object will be on this same trajectory, at point B.

A Kalman filter gives us a mathematical way to infer velocity from only a set of measured locations. In this lesson, we'll learn how to create a 1D Kalman filter that takes in positions, like those shown above, takes into account uncertainty, and estimates where future locations might be and the velocity of an object!

In Kalman filters, we iterate measurement and motion is often called the measurement update and prediction. The outcome use Bayes Rule, which is called the product. The outcome is total probability which is called cololution or addition. 
