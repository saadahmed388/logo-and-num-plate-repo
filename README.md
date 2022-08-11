# logo and num plate repo
In this project tensorflow object detecting module is used to train the object detection model using RESNET 640 x 640 architecture, the dataset was self collected and was annotated for logo and number plate using labelimg tool
Used a CNN for car logo brand prediction (logo brand prediction is limited to these 8 brands namely "Hyundai","Lexus","Mercedes","Opel","Skoda","Toyota","Volkswagen","Mazda" due to lack of dataset), the dataset was trained on Xception CNN architecture which is a slightly advanced version of inception V3.Training on this model resulted in a validation accuracy of 86%.
For the number plate part, easyocr library is used to produce string output by reading the number plate.

Further more, the model for detection on images is deployed using the gradio interface.
The detection on images can be done using the detection-image.py file and for performing detection using webcam or videofile upload can be done using detection-vid.py file (For the video part, when the frame window opens after running the code, press 'q' for performing the prediction, pressing 'a' breaks the operation).

On running detection-to-video.py file (Input method be video file or webcam) an avi video file named 'detected.avi' will be saved on a fps rate of 20 in the working directory (pressing 'a' would stop the saving)



