import numpy as np
import tensorflow as tf
import cv2
import os

# We are importing the model 
model = tf.keras.models.load_model('model.keras')

#We will load the image that we want to predict 
#for that we will specify a list that contain the images that the model can predict
#the names of crop are arranged in order
crop_names = ['jute', 'maize', 'rice', 'sugarcane', 'wheat']# Don't Change the order of the list
#we will load the image now
image_size = 200 #This is fixed value, changing this value will make the model disfunction
image_path = os.path.join('test_crop_image/jute002.jpg')#Replace the path with yout Path
image = cv2.imread(image_path)
image = cv2.resize(image, (image_size, image_size))
image = np.array([image]).reshape(-1, image_size, image_size, 3)# 3 is used because model is trained on coloured image

#we have to covert the array to float
prediction = model.predict(image/255.0)
print(crop_names[np.argmax(prediction)])