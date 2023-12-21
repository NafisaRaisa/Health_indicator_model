

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score #cross validation
from sklearn.model_selection import train_test_split #split the available dataset for training and testing

import io
from google.colab import files
uploaded = files.upload() #upload csv file from computer
df= pd.read_csv(io.BytesIO(uploaded['diabetes_binary_5050split_health_indicators_BRFSS2015.csv']))
uploaded.keys()

print(df.head())

df.info()

print(df.columns)

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
columns = ["BMI", "HighBP", "GenHlth", "PhysHlth", "HighChol", "Age", "DiffWalk", "Income", "Education", "PhysActivity", "Diabetes_binary"] # 10 best features
df = df[columns]
print("Contents in csv file:\n", df)
df.to_csv("file with 10 features.csv", index=False)
plt.show()

df =pd.read_csv("file with 10 features.csv")
df.info()

# Convert to object data types
df["HighBP"] = df["HighBP"].astype('object')
df["HighChol"] = df["HighChol"].astype('object')
df["BMI"] = df["BMI"].astype('object')
df["GenHlth"] = df["GenHlth"].astype('object')
df["PhysHlth"] = df["PhysHlth"].astype('object')
df["DiffWalk"] = df["DiffWalk"].astype('object')
df["Age"] = df["Age"].astype('object')
df["Education"] = df["Education"].astype('object')
df["Income"] = df["Income"].astype('object')
df["PhysActivity"] = df["PhysActivity"].astype('object')

# Split the dataset into features and labels
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

# Split data into training, validation, and testing sets, as well as stratify it
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.33, random_state=0, stratify=y_val_test)

X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape

# create a decision tree classifier
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=25, criterion='entropy')
dt.fit(X_train, y_train)

# Perform cross-validation
scores = cross_val_score(dt, X_train, y_train, cv=15, scoring='accuracy')
print("Cross-validation scores:", scores)
print("Average score:", np.mean(scores))

# Make predictions on test set
y_pred = dt.predict(X_test)

# Calculate accuracy of the model on test set
accuracy = np.mean(y_pred == y_test)
print("Accuracy on test set:", accuracy)

from sklearn.metrics import f1_score, precision_score, recall_score

# calculate F1 score
f1 = f1_score(y_test, y_pred)

# calculate precision
precision = precision_score(y_test, y_pred)

# calculate recall
recall = recall_score(y_test, y_pred)

print("F1 score:", f1)
print("Precision:", precision)
print("Recall:", recall)

#formulate the decision tree graphic
from IPython.display import Image
from io import StringIO
import pydotplus

dot_data = StringIO()
export_graphviz(dt, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=X_train.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_jpg())
from google.colab import files

#download to local computer
with open("decision_tree.jpg", "wb") as f:
    f.write(graph.create_jpg())
files.download("decision_tree.jpg")