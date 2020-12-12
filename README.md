# 2020F EE551-WS Final Project

##Keywords
binary classification problem, data analysis, machine learning

##Problem
 
Location awareness is a key feature for many upcoming application scenarios, e.g., asset tracking, autonomous navigation or ambient assisted living. For indoor environments, ultra-wideband (UWB) technologies have been proposed due to the superior time-resolution. However, the environmental characteristic of the positioning site is complex and dynamic. The direct UWB signals might be obstructed, which is called non-line-of-sight (NLOS) case. NLOS propagation is the main error source of UWB ranging and positioning system.

NLOS identification attempts to distinguish between LOS and NLOS conditions. And in mathematics, it is a binary classification problem, which can be solved by machine learning.


## Solution
First, we find an open-source data set on the GitHub (https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set). NLOS and LOS measurements were collected from seven different indoor locations: Office1, Office2, Small apartment, Small workshop, Kitchen with a living room, Bedroom and Boiler room. At each location, 3000 NLOS and 3000 LOS measurements were collected.

Second, we use Python pandas library to read, describe, and analyze the distributions of the data set. And Python seaborn library is used to plot the violin chart of each variable.

Third, we use Python sklearn library to conduct the linear regression, decision tree, support vector machine to classify the LOS and NLOS cases.
Finally, we calculate the prediction accuracy and execution time of each algorithm, and use Python matplotlib library to plot them on figures.


## Highlights
* Solve a binary classification problem assisted with Python libraries.
* Use Python to visualize the distributions of each variable.
* Compare the prediction accuracy and execution time of each machine learning algorithm.


##Appendix: Data source introduction

Data source from here: https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set

The data set was created using SNPN-UWB board with DecaWave DWM1000 UWB radio module.

Measurements were taken on 7 different indoor locations:

- Office1
- Office2
- Small appartment
- Small workshop
- Kitchen with a living room
- Bedroom
- Boiler room

In every indoor location 3000 LOS samples and 3000 NLOS samples were taken. Different locations were choosen to prevent building of location-specific LOS and NLOS models. All together 42000 samples were taken: 21000 for LOS and 21000 for NLOS channel condition. To make data set ready for building LOS and NLOS models, samples are randomized to prevent overfitting of a model to particular places. For measurements two UWB nodes were used: one node as an anchor and the second node as a tag. Only traces of LOS and NLOS channel measurements were taken without any reference positioning (this data set is not appropriate for localization evaluation).