from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from google.colab import files
import io

# uploading the dataset and saving it
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['diabetes_binary_5050split_health_indicators_BRFSS2015.csv']))

# picking 10 top features based on experimenting with the accuarcy and looking at the correlations
# and then make a file only with those 10 features and the class label
columns = ["Diabetes_binary","BMI", "HighBP", "PhysActivity", "GenHlth", "PhysHlth", "Age", "Education", "Income", "DiffWalk", "HighChol"]
df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv", usecols=columns)
print("Contents in csv file:\n", df)

df.info()

#performing k-neighbors on our dataset
# Split your dataset into training and testing sets
X = df.drop("Diabetes_binary", axis=1)  # class label
y = df["Diabetes_binary"]  # target variable

# Assuming you have your data in X and labels in y

# Split the data into training, test, and evaluation sets
X_train_test, X_eval, y_train_test, y_eval = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=0.22, random_state=42)

# Define the range of k values to evaluate
k_values = [1,2,3,4,5,6,7,8,9,10]

# Perform k-fold cross-validation on the evaluation set for each k value
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the k-NN model on the combined training and testing set (equal to 80% of the dataset)
    knn.fit(X_train_test, y_train_test)

    # Make predictions on the evaluation set (10% of the dataset)
    y_pred = knn.predict(X_eval)

    # Step 6: Evaluate the model using cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    precision_scores = cross_val_score(knn, X, y, cv=cv, scoring='precision')
    recall_scores = cross_val_score(knn, X, y, cv=cv, scoring='recall')
    accuracy_scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')

    # Step 7: Report the mean scores
    print("For K value:"+ str(k))
    print("Mean precision score:", np.mean(precision_scores))
    print("Mean recall score:", np.mean(recall_scores))
    print("Mean accuracy score:", np.mean(accuracy_scores))

  
