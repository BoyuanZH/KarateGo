##KarateGo
>Author: BoyuanZH
>
>KarateGo is a educational and entertaining program which utilizes machine learning algorithm to provide customized guidance for users to do basic karate techniques, such as middle punches, round kicks etc.
>
>The program takes advantage of the the body frame and color frame generated through kinect sensor to get the user's skeleton information. The skeleton information can be further processed by the program. Machine learning is used here to do gesture recognition, that is to determine the technique the user is doing. To control the system, the program utilizes built-in hand-gesture recognizing function of pykinect2 module. Pygame is used here to support the game frame and game logic.


###Installation Guide

>All project source files needed to run the KarateGo can be downloaded here. Design_Document folder is included for illustrating design logic.


1.    **Packages and Software**

      Please follow [this instructions](https://onedrive.live.com/view.aspx?cid=ed75cbdc5e4ab0fe&page=view&resid=ED75CBDC5E4AB0FE!1302823&parId=ED75CBDC5E4AB0FE!1096749&app=PowerPoint) to install all the modules that are needed to run microsoft kinect on PC. Visual Studio 2017 or 2015 version is required.

      Also make sure you have required packages installed, including pygame, sklearn, PyKinect2, numpy.
      
2.    **Devices**

      One Microsoft Kinect and one PC are required to run the program.

3.    **Run the GarateGO**

      After all the devices and modules are installed sucessfully, you can download the whole project file and run GarateGo in Microsoft Visual Studio (2015 or 2017), by openning the solution file named ***KarateGo.sln*** in the outter most KarateGo folder.