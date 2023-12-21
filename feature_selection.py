

df.info()                               #gets info on dfset

pd.read_csv('/content/drive/MyDrive/Colab Notebooks/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

import pandas as pd
                                                        # Make sure pandas is loaded
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

#location of the datatse
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/diabetes_binary_5050split_health_indicators_BRFSS2015.csv') # Note that pd.read_csv is used because we imported pandas as pd

#change data to int
df["Diabetes_binary"] = df["Diabetes_binary"].astype('int')
df["HighBP"] = df["HighBP"].astype('int')
df["HighChol"] = df["HighChol"].astype('int')
df["CholCheck"] = df["CholCheck"].astype('int')
df["BMI"] = df["BMI"].astype('int')
df["Smoker"] = df["Smoker"].astype('int')
df["Stroke"] = df["Stroke"].astype('int')
df["HeartDiseaseorAttack"] = df["HeartDiseaseorAttack"].astype('int')
df["PhysActivity"] = df["PhysActivity"].astype('int')
df["Fruits"] = df["Fruits"].astype('int')
df["Veggies"] = df["Veggies"].astype('int')
df["HvyAlcoholConsump"] = df["HvyAlcoholConsump"].astype('int')
df["AnyHealthcare"] = df["AnyHealthcare"].astype('int')
df["NoDocbcCost"] = df["NoDocbcCost"].astype('int')
df["GenHlth"] = df["GenHlth"].astype('int')
df["MentHlth"] = df["MentHlth"].astype('int')
df["PhysHlth"] = df["PhysHlth"].astype('int')
df["DiffWalk"] = df["DiffWalk"].astype('int')
df["Sex"] = df["Sex"].astype('int')
df["Age"] = df["Age"].astype('int')
df["Education"] = df["Education"].astype('int')
df["Income"] =df["Income"].astype('int')

df.info()

df.size                                       #the total number of array elements

df.shape                                          #the size of each dimension

df.head()

df.describe(include = 'all')

df.apply(pd.Series.value_counts)

for column in df.columns:
    print(column)
    print(df[column].value_counts().sort_index())
    print()

#print out top 20 most popular bmi
print(df['BMI'].value_counts().sort_values(ascending=False).head(20))

for col in df.columns:
    # Get a count of the occurrences of each unique value in the column
    value_counts = df[col].value_counts()

    # Plot a bar chart for the value counts of the column
    ax = value_counts.plot(kind='bar', figsize=(15, 6))
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    plt.show()

#printing out bmi bar graph with top 45 for slides
# Get a count of the occurrences of each unique value in the BMI column
value_counts = df['BMI'].value_counts().head(45)

# Plot a bar chart for the value counts of the BMI column
ax = value_counts.plot(kind='bar', figsize=(15, 6))
ax.set_xlabel('BMI')
ax.set_ylabel('Count')
plt.show()

df.corr()  #check correlation

corr_matrix = df.corr()
print(corr_matrix['Diabetes_binary'])       #only prints corr between Diabetes_binary and rest

abs_corr = abs(df.corr(numeric_only=False)['Diabetes_binary'])
top15_features = abs_corr.sort_values(ascending=True)[1:16].index.tolist()
print(top15_features)

abs_corr = abs(df.corr(numeric_only=False)['Diabetes_binary'])
top15_features = abs_corr.sort_values(ascending=False)[1:16]

print("Top 15 features correlated with Diabetes_binary:")
for feature, corr in top15_features.items():
    print(f"{feature} : {corr:.3f}")

# Print unique values for columns
print('Unique values in column CholCheck:', df['CholCheck'].unique())
print('Unique values in column BMI:', df['BMI'].unique())
print('Unique values in column Smoker:', df['Smoker'].unique())
print('Unique values in column Stroke:', df['Stroke'].unique())
print('Unique values in column HeartDiseaseorAttack:', df['HeartDiseaseorAttack'].unique())
print('Unique values in column PhysActivity:', df['PhysActivity'].unique())
print('Unique values in column Fruits:', df['Fruits'].unique())
print('Unique values in column Veggies:', df['Veggies'].unique())
print('Unique values in column HvyAlcoholConsump:', df['HvyAlcoholConsump'].unique())
print('Unique values in column AnyHealthcare:', df['AnyHealthcare'].unique())
print('Unique values in column NoDocbcCost:', df['NoDocbcCost'].unique())
print('Unique values in column GenHlth:', df['GenHlth'].unique())
print('Unique values in column MentHlth:', df['MentHlth'].unique())
print('Unique values in column PhysHlth:', df['PhysHlth'].unique())
print('Unique values in column DiffWalk:', df['DiffWalk'].unique())
print('Unique values in column Sex:', df['Sex'].unique())
print('Unique values in column Age:', df['Age'].unique())
print('Unique values in column Education:', df['Education'].unique())
print('Unique values in column Income:', df['Income'].unique())

"""Observation:
GenHlth and PhysHlth have postive relation with each other.
GenHlth and Income have negative relation with each other.

Observation: There is a positive corelation between Diabetes_binary and genhealth,highbp,bmi and difwalk. There is a negative correlation between diabete_binary and income,education and physical activty.
"""

df.isnull().sum() #check null vals

plt.figure(figsize = (20,10))
sns.heatmap(df.corr(),annot=True , cmap ='coolwarm' )
plt.title("correlation of all 22 features")

df[['Diabetes_binary', 'HighBP', 'HighChol', 'BMI', 'HeartDiseaseorAttack', 'GenHlth', 'PhysHlth', 'DiffWalk', 'Age', 'Income']].hist(figsize=(20,10))
plt.tight_layout()
plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# separate the target variable and the features
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

# select the k best features using chi-square test
selector = SelectKBest(chi2, k=10)
X_new = selector.fit_transform(X, y)

# get the scores and p-values of the features
scores = selector.scores_
pvalues = selector.pvalues_

# print the results
for i in range(len(X.columns)):
    print(f"{X.columns[i]}: score={scores[i]}, p-value={pvalues[i]}")

# Select top 15 features using chi-square test
top_15_features = SelectKBest(chi2, k=15).fit(X, y)

# Print top 15 features based on chi-square score
feature_scores = pd.DataFrame({'feature': X.columns, 'score': top_15_features.scores_})
top_15 = feature_scores.nlargest(15, 'score')
print(top_15)




#top 15 heatmap

corr_matrix = df[['Diabetes_binary', 'GenHlth', 'HighBP', 'DiffWalk', 'BMI', 'HighChol', 'Age', 'HeartDiseaseorAttack', 'PhysHlth', 'Income', 'Education', 'PhysActivity', 'Stroke', 'MentHlth', 'CholCheck', 'Smoker']].corr()

plt.figure(figsize = (20,10))
sns.heatmap(corr_matrix,annot=True , cmap ='coolwarm' )
plt.title("correlation of top 15 features")
