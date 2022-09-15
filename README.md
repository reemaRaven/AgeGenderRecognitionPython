# AgeGenderRecognitionPython
A tensorflow backend based solution that crops real time face images from webcam , and identifies gender as well as age of the person

# Pre-requisites
1. Install python 3.6 or higher (as tensorflow image-processing backend works with Python 3 or higher)
2. Install :  numpy 1.13.3+mkl , Keras 2.0.8+ , TensorFlow 1.4.0 , opencv 1.0.1+
(process to individually install these : "python -m pip install tensorflow" etc. , when installing from the shell)
3. Create a weights.18-4.06 folder in this directory and keep the weights downloaded file from here : 
https://drive.google.com/file/d/1_t6_T3bo-cLHemX7ZQ6yxF6lfDrKDNvw/view?usp=sharing

# Running the demo
1. Run the main file : "python realtime_demo.py"
2. Below is actual demo :

https://drive.google.com/file/d/13albCqk1Rm2nagHeMMVx38F2NjW0Zibh/view?usp=sharing

# Description of project
1. Photo per second is taken from the webcam stream live by the cv2 module.
2. Image is turned into grayscale and use the CascadeClassifier class to detect faces in the image
3. Variable faces return by the detectMultiScale method is a list of detected face coordinates [x, y, w, h].
4. Next is to crop those faces before feeding to the neural network model, after adding 40% margin to the face area so that the full head is included.
5. Feed those cropped faces to the model, by calling the predict method. 
6. Age prediction : the output of the model is a list of 101 values associated with age probabilities ranging from 0~100, and all the 101 values add up to 1 (softmax). Then multiply each value with its associated age , add them up resulting in final predicted age.
7. Gender prediction : its a binary classification task. The model outputs value between 0~1, where the higher the value, the more confidence the model think the face is a male.
8. Finally, draw the result and render the image. 
