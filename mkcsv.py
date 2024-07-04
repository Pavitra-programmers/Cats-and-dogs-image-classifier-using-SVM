import tensorflow as tf
import numpy as np
import pandas as pd
from keras.api.applications.resnet50 import ResNet50, preprocess_input
from keras.api.preprocessing import image
import os
import pickle

  # Load the ResNet50 model pretrained on ImageNet
model = ResNet50(weights='imagenet', include_top=False)

  # Path to the Kaggle Dogs vs. Cats dataset
dataset_path = 'your dataset'

  # Function to extract features from an image
def extract_features(img_path):
      img = image.load_img(img_path, target_size=(224, 224))  # Resizing the image to match the input size of ResNet50
      img = image.img_to_array(img)  # Convert image to numpy array
      img = np.expand_dims(img, axis=0)  # Add an extra dimension to represent batch size
      img = preprocess_input(img)  # Preprocess the image (e.g., normalization)

      features = model.predict(img)  # Extract features using ResNet50
      features = features.flatten()  # Flatten the feature tensor to a 1D vector

      return features

  # List of image paths

labels = []
features_list = []
i=0
  #Extracting the features of the image and thier labels
for file in os.listdir(dataset_path):         
    img_path = os.path.join(dataset_path, file)
    features = extract_features(img_path)#using the above function
    features_list.append(features)
    if file[0] == 'd':#labeling our data
      labels.append(1)
    else:
      labels.append(0)
  
  # Convert the list of feature vectors and labels to a pandas DataFrame
df = pd.DataFrame(features_list)
df1 = pd.DataFrame(labels, columns=['label'])
df2 = pd.concat((df,df1), axis=1)
print(df2[0:5])
df2.to_csv('image_features.csv',index=False)


