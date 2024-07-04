import tensorflow as tf
from keras.api.applications.resnet50 import ResNet50, preprocess_input
from keras.api.preprocessing import image
import numpy as np
import pandas as pd
import pickle

# Load the ResNet50 model pretrained on ImageNet
model = ResNet50(weights='imagenet', include_top=False)
#taking the input of the image
img = input('Enter the path of the image :- \n')
#defining the function for extracting the features of the image
def extract_features(img_path):
      img = image.load_img(img_path, target_size=(224, 224))  # Resizing the image to match the input size of ResNet50
      img = image.img_to_array(img)  # Convert image to numpy array
      img = np.expand_dims(img, axis=0)  # Add an extra dimension to represent batch size
      img = preprocess_input(img)  # Preprocess the image (e.g., normalization)

      features = model.predict(img)  # Extract features using ResNet50
      features = features.flatten()  # Flatten the feature tensor to a 1D vector
      return features
feture_list = []
features = extract_features(img)#extracting the features using above function
feture_list.append(features)
df = pd.DataFrame(feture_list)#transforming it into pandas dataframe
#loading the saved model
pick = open('Model.sav', 'rb')
model = pickle.load(pick)
#prediction
res = model.predict(df)

if res[0] ==1 :
      print('Its a Dog.')
else:
      print('Its a Cat.')