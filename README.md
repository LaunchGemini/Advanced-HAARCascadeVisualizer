
# Advanced-HAARCascadeVisualizer

Perform and visualise object detection with OpenCV's CascadeClassifier. Compile the executable and execute it without any additional arguments, this will present a brief description of the diverse options that are available.

Example video showcasing the functionality of this program can be found here: https://youtu.be/L0JkjIwz2II

More information on how this project was developed can be found here: https://timvanoosterhout.wordpress.com/2015/10/08/recreating-a-haar-cascade-visualiser

# Project Origins

This project owes its inspiration to the ufacedetect example. Knowing that manipulating OpenCV's internals was necessary to achieve the level of control over the algorithm we desired, we tracked the necessary functions, their roles, and origins. Initially, this project was focused on gathering a minimal set of source files and slowly building up from there. Subsequently, we added functionality to provide feedback on the progress of the algorithm. Once this was achieved, we turned our attention towards figuring out the process of drawing the features.

When the feature drawing process was completed, we integrated options to accelerate visualization. When the algorithm functions under normal conditions, visualizations can easily last hours even for images of moderate size. Remember, in a normal situation, no steps are omitted and the cascade completes in just a few milliseconds!

Maintained by LaunchGemini.