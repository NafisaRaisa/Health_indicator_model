# Health_indicator_model

The dataset being used is related to diabetes and its associated risk factors. This dataset was a health-related telephone survey collected annually by the CDC. Interviewers collect data from a randomly selected adult in a household (through telephone interviews using random-digit-dialing (RDD)) Each year, the survey collects responses from over 400,000 Americans on health-related risk behaviors, chronic health conditions, and the use of preventative services. It has been conducted every year since 1984. For this project, we are using data from the 2015 survey. The dataset has 22 features, along with a class label that indicates whether someone has diabetes or not.

The goal of the project is to identify the best features or indicators of diabetes out of the 22 features available in the original dataset. We use feature selection methods and then implement different classification models like Naive-Bayes, Decision-Tree, Random-Forests, and K-Neighbors to see what model best predicts diabetes.

Data Cleaning: The original dataset needed some values and needed to be more balanced. 80% of people did not have diabetes, and only 20% did. Therefore, the dataset must be cleaned and balanced before feature selection or classification models can be performed. There was a version of the dataset available in kaggle.com that was already cleaned but not balanced. Using the cleaned but unbalanced dataset called “diabetes_binary_health_indicators_BRFSS2015.csv", balanced the dataset and generated a new dataset called “diabetes_binary_5050split_health_indicators_BRFSS2015.csv”. The “diabetes_binary_5050split_health_indicators_BRFSS2015.csv” file was used for the rest of the project.

Implemented different machine learnibg models. Below are the results:


|              | Naive-Bayes   | Random-Forest | Decision-Tree | K-Neighbors |
|--------------|---------------|---------------|---------------|-------------|
| Precision    | 73%           | 70.02%        | 70.99%        | 74.21%      |
| Recall       | 74.08%        | 68.63%        | 80.52%        | 49.25%      |
| Accuracy     | 73%           | 69.59%        | 72.59%        | 66.21%      |






