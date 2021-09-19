# ImuTracker

This repository provides python classes and scripts to process data of inertial measurement units (acceleration, rotation rate and magnetic field) to obtain rotation and position data of the tracked object.
The framework was developed to reconstruct movement data of pedestrians in crowds. 
The basis for the tracking systems is a 2D camera system (overhead recordings). 
In addition, IMU data can be used to capture rotation information of the upper body or to reconstruct trajectories in case of occlusion.

The framework offers methods to: 
- fuse camera and IMU data
- track the rotation of the upper body
- track the position of homogeneously moving pedestrians (in this case wheelchair users)
- validate the tracking techniques
- visualize the data



## Demo
An example for the calculation of rotation angles of pedestrians based on trajectory files and extended by IMU data can be found in the `demo` directory.
Configuration files for basic parameters and algorithm specific setting are also provided there.
