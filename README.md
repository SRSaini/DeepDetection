# DeepDetection

Classification of a Fender Apronas defective and non-defective using Transfer Learning with f-1 score of 1.00

Contents

functions.py contains functions for preprocessing of images and making classes
routes.py is the Flask api file
emplates contain index.html and submit.html, the frontend of the application

obileNet_model.json & MobileNet_model_wieghts.h5 are saved model and its weights respectively, which are deployed in our application


The data is already labelled having a total of 250 images with 139 images as healthy machine parts and rest 111 as defective parts.
Images given in the dataset were captured from different angles and scales. Training and Test datasets were prepared by randomly 
selecting a total of 25 images (i.e 10%) in which 12 were defective and 13 were healthy parts. Training/validation split used is 90/10.

