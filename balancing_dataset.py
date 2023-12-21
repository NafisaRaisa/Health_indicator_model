from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd

# Assuming your dataset is stored in a pandas DataFrame
# X denotes the feature matrix, and y denotes the target variable

# Load your dataset into X and y
X = pd.read_csv('https://ibrahimnabid.github.io/diabetes_binary_health_indicators_BRFSS2015.csv')
y = X['Diabetes_binary']
X = X.drop('Diabetes_binary', axis=1)

# Instantiate the RandomOverSampler
oversampler = RandomOverSampler(random_state=42)

# Resample the dataset
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Print the class distribution before and after balancing
print("Class distribution before balancing:")
print(y.value_counts())

print("Class distribution after balancing:")
print(np.bincount(y_resampled))

df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_balanced['Diabetes_binary'] = y_resampled

# Save the balanced dataset to a CSV file with a different name
df_balanced.to_csv('diabetes_binary_5050split_health_indicators_BRFSS…', index=False)

# Now you can use the balanced dataset (X_resampled and y_resampled) for further analysis or modeling
