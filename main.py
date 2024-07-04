import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

  # Load the extracted features from the CSV file
df = pd.read_csv('image_features.csv')
y = df['label']
df = df.drop('label', axis=1)

 # Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
      df,y, test_size=0.3, random_state=42
  )


  # Initialize and train the SVM model with hyperparameters
svm_model = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

  # Make predictions on the test set
y_pred = svm_model.predict(X_test)

#saving the model
with open('Model.sav','wb') as pick:
  pickle.dump(svm_model,pick)
  
  # Print classification report
print("Classification Report:",classification_report(y_test, y_pred))

  # Print accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))

  # Print confusion matrix
print("Confusion Matrix:",confusion_matrix(y_test, y_pred))

