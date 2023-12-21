import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import io
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from google.colab import files

#uploading the dataset as it is and saving it
from google.colab import files
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['diabetes_binary_5050split_health_indicators_BRFSS2015.csv']))
