#library import and dataset upload
import sklearn.naive_bayes
from sklearn.metrics import precision_score, recall_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import io
from google.colab import files
uploaded = files.upload()
df= pd.read_csv(io.BytesIO(uploaded['diabetes5050.csv']))
df.info()

#change dtype for BMI based on arbitrary categories
  #0=underweight
  #1=normal weight
  #2=overweight
  #3=obese
#df['BMI']=df['BMI'].astype('float')
bins=[0,18.5,25,30,100]
labels=[0,1,2,3]
df['BMI']=pd.cut(df['BMI'], bins=bins, labels=labels)
#df['replace_BMI']

#remove original BMI category
#df=df.drop('BMI',axis=1, inplace=True)


###naive bayes classification using all features
X = df.iloc[:,1:22]
X.head()

Diabetes_Category = {} # a dictionary with key-value pairs 
Diabetes_Category['feature_names'] = X.columns.values

enc = OrdinalEncoder()
X = enc.fit_transform(X)
X

Diabetes_Category['data']=X

Y = df.iloc[:,0:1]
Diabetes_Category['target_names']=Y['Diabetes_binary'].unique()
Diabetes_Category['target']=Y["Diabetes_binary"].values

Diabetes_Category['target_names']
Diabetes_Category['target']

NB_C = CategoricalNB()

#splitting
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

scores_NBC = cross_val_score(NB_C, Diabetes_Category['data'], Diabetes_Category['target'], cv=5, scoring='accuracy')
print(scores_NBC)

#mean and 95% confidence level
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_NBC.mean(), scores_NBC.std() * 2))

y_predict= NB_C.fit(Diabetes_Category['data'],Diabetes_Category['target']).predict(Diabetes_Category['data'])
y_pred_test=NB_C.predict(X_test)
y_pred_eval=NB_C.predict(X_eval)

#precision score
precision_test=precision_score(y_test, y_pred_test)
precision_eval=precision_score(y_eval,y_pred_eval)
print("Precision for Testing:", precision_eval)
print("Precision for Evaluation:", precision_eval)


#recall score
recall_test=recall_score(y_test, y_pred_test)
print("Recall:", recall_test)


df=df.iloc[0:70692]
df['predict_all']= y_predict
print(df)

print(confusion_matrix(Diabetes_Category['target'], y_predict))
