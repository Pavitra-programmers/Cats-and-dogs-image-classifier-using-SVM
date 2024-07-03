import tensorflow as tf
from keras.api.applications.resnet50 import ResNet50, preprocess_input
from keras.api.preprocessing import image
import numpy as np
import pandas as pd
import pickle


model = ResNet50(weights='imagenet', include_top=False)

img = input('Enter the path of the image :- \n')

def extract_features(img_path):
      img = image.load_img(img_path, target_size=(224, 224))  # Resizing the image to match the input size of ResNet50
      img = image.img_to_array(img)  # Convert image to numpy array
      img = np.expand_dims(img, axis=0)  # Add an extra dimension to represent batch size
      img = preprocess_input(img)  # Preprocess the image (e.g., normalization)

      features = model.predict(img)  # Extract features using ResNet50
      features = features.flatten()  # Flatten the feature tensor to a 1D vector
      return features
feture_list = []
features = extract_features(img)
print(features)
feture_list.append(features)
# df = pd.DataFrame(feture_list)
pick = open('Model.sav', 'rb')
model = pickle.load(pick)

res = model.predict(feture_list)

# print(res)