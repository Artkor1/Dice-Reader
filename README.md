# Dice-Reader

Dice Reader is an application that uses Deep Neural Network models to perform object detection and predict the results on images with multiple dice.

Machine learning part was done in Google Colaboratory notebook, which allowed me to use powerful graphical processing units: Tesla K80, Tesla T4. I used pre-trained Faster R-CNN ResNet-50 FPN architecture from PyTorch model zoo. Train (82%) and validation (18%) data sets contain 916 images of multiple dice each (2682 dice in total). Dice include: six-sided, eight-sided, ten-sided and twelve-sided dice of different size, color, shape, etc. 4 models of varying accuracy were created using different data augmentations. Some of the images come from Kaggle datasets, but most of them were mede by me using my personal dice collection. Aside from PyTorch framework I used torchvision, Tensorflow and detecto libraries for machine learning as well as matplotlib for visualization. I also used LabelImg tool to label the images.

Application allows users to predit dice results with selected model and threshold (0.80 by default). Image with marked results can be saved to file. Labels, prediction scores and boxes can be turned on/off. Additionaly: Total sum, average dice result and other information can be found below the image. I used Python 3.9 to create the application. Graphical interface was made using PyQt5 library and OpenCV helped me work with images. 

"project.png" and "project2.png" show the application in action. Accuracy of the models is satisfying, although there are certain problems with more complex images. For example: reading multiple visible sides of one dice instead of just the top side.

Trained models and the dataset are not included in the repository, because of their sheer size and volume.

