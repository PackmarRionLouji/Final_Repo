import numpy as np
from tensorflow.keras.models import load_model
import io
from PIL import Image
from keras_preprocessing import image
from flask import Flask, jsonify, request
import cv2
import pandas as pd
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    filename = file.filename
    file.save(filename)
    model = load_model('pneumonia_detection_model (1).h5')
    test_image = image.load_img(filename, target_size = (256,256))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    prediction = model.predict(test_image)[0][0]
    threshold = 0.5 # Set threshold for predicting pneumonia
    if prediction > threshold:
        outl= 'Traces of PNEUMONIA is present.'
    else:
        outl= 'Traces of PNEUMONIA is not present.'
    
    out = prediction.tolist()
    
    img = cv2.imread(filename)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    lung_pixels = np.sum(markers == 2)
    total_pixels = np.prod(markers.shape)
    lung_percentage = lung_pixels / total_pixels * 100
    output=str(lung_percentage)
    def lungper(x):
        a=round(x,2)
        return f"{a:.2f}% of lungs is affected"
    def outpu(x):
        return str(x)
    print(outpu(outl))
    print(lungper(lung_percentage))
    return [outpu(outl),lungper(lung_percentage)]

    #print('Percentage of lungs affected: {:.2f}%'.format(lung_percentage))
    #return [str(out),output]
if __name__ == '__main__':
    app.run(debug=True)
