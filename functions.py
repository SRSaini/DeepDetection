import pandas as pd
from PIL import Image  # pillow
import cv2


def pre_processing(image_path):
    """
     Function performs minor processing of rotation, blurring, resizing and grayscale conversion and returns tuple containing 
     resized gray, blurred and original images
    """

    #training_data = []
    import cv2  # openCV
    import numpy as np
    # Reading the image with opencv

    image = cv2.imread(image_path)
    image = np.array(image, dtype=np.uint8)

    #img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_CUBIC)
    # training_data.append([np.array(img)])
    # changing to grayscale

    # list to array
    #X = np.array(training_data)

    return image


def make_classes(y_pred):
    """
    Function takes in the prediction array from the model and gives classes of "Defective" and "Healthy" to the results
    along with the probability associated with our prediction in form of a tuple.
    """
    for i in y_pred:
        if i[0] > 0.5:
            return "Healthy", i[0]
        elif i[0] <= 0.5:
            return "Defective", i[0]


def pred(test_image_path):
    """
    Main function for image prediction which uses saved MobileNet model to return resulted class using make_classes function
    """
    import keras
    from keras.applications import MobileNet
    from keras import optimizers
    from keras.models import load_model
    from tensorflow.keras.models import model_from_json
    import cv2
    import numpy as np
    from PIL import Image  # pillow
    from functions import pre_processing, make_classes
    from keras.utils.generic_utils import CustomObjectScope
    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True
    import tensorflow as tf

    with open("MobileNet_model.json") as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        # print(loaded_model.summary())
        # load weights into new model
        loaded_model.load_weights("MobileNet_model_wieghts.h5")
        sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
        loaded_model.compile(loss='binary_crossentropy',
                             optimizer=sgd,
                             metrics=['accuracy'])

        X_test = []
        X_test.append(cv2.resize(pre_processing(test_image_path),
                                 (600, 600), interpolation=cv2.INTER_CUBIC))
        img = np.array(X_test)
        pred = loaded_model.predict(img)
    return make_classes(pred)
