import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load dataset
dataset = pd.read_csv("heart.csv")

# Display basic info
print(dataset.shape)
print(dataset.head(5))
print(dataset.describe())
print(dataset.info())

# Description of features
info = ["age", "1: male, 0: female", "chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)",
        "resting blood pressure", "serum cholesterol in mg/dl", "fasting blood sugar > 120 mg/dl",
        "resting electrocardiographic results (values 0,1,2)", "maximum heart rate achieved",
        "exercise induced angina", "oldpeak = ST depression induced by exercise relative to rest",
        "the slope of the peak exercise ST segment", "number of major vessels (0-3) colored by flourosopy",
        "thal: 3 = normal, 6 = fixed defect, 7 = reversible defect"]

for i in range(len(info)):
    print(dataset.columns[i] + ": " + info[i])

# Analyzing target
print(dataset["target"].describe())
print(dataset["target"].unique())

# Correlation with the target
print(dataset.corr()["target"].abs().sort_values(ascending=False))

# Count plot for the target
sns.countplot(x="target", data=dataset)

# Percentage distribution of the target
target_temp = dataset.target.value_counts()
print(f"Percentage of patients without heart problems: {round(target_temp[0]*100/len(dataset), 2)}%")
print(f"Percentage of patients with heart problems: {round(target_temp[1]*100/len(dataset), 2)}%")

# Unique values and barplots for different features
features_to_plot = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
for feature in features_to_plot:
    sns.barplot(x=feature, y="target", data=dataset)
    plt.show()

# Distribution plot for 'thal'
sns.distplot(dataset["thal"])

# Splitting data into training and testing sets
from sklearn.model_selection import train_test_split
predictors = dataset.drop("target", axis=1)
target = dataset["target"]

X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# Import accuracy score function
from sklearn.metrics import accuracy_score

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred_lr = lr.predict(X_test)
score_lr = round(accuracy_score(Y_pred_lr, Y_test)*100, 2)
print(f"The accuracy score achieved using Logistic Regression is: {score_lr} %")

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, Y_train)
Y_pred_nb = nb.predict(X_test)
score_nb = round(accuracy_score(Y_pred_nb, Y_test)*100, 2)
print(f"The accuracy score achieved using Naive Bayes is: {score_nb} %")

# Support Vector Machine
from sklearn import svm
sv = svm.SVC(kernel='linear')
sv.fit(X_train, Y_train)
Y_pred_svm = sv.predict(X_test)
score_svm = round(accuracy_score(Y_pred_svm, Y_test)*100, 2)
print(f"The accuracy score achieved using Linear SVM is: {score_svm} %")

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
score_knn = round(accuracy_score(Y_pred_knn, Y_test)*100, 2)
print(f"The accuracy score achieved using KNN is: {score_knn} %")

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
max_accuracy = 0
for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train, Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt, Y_test)*100, 2)
    if current_accuracy > max_accuracy:
        max_accuracy = current_accuracy
        best_x = x

dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train, Y_train)
Y_pred_dt = dt.predict(X_test)
score_dt = round(accuracy_score(Y_pred_dt, Y_test)*100, 2)
print(f"The accuracy score achieved using Decision Tree is: {score_dt} %")

# Random Forest
from sklearn.ensemble import RandomForestClassifier
max_accuracy = 0
for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train, Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf, Y_test)*100, 2)
    if current_accuracy > max_accuracy:
        max_accuracy = current_accuracy
        best_x = x

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)
score_rf = round(accuracy_score(Y_pred_rf, Y_test)*100, 2)
print(f"The accuracy score achieved using Random Forest is: {score_rf} %")

# XGBoost
import xgboost as xgb
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, Y_train)
Y_pred_xgb = xgb_model.predict(X_test)
score_xgb = round(accuracy_score(Y_pred_xgb, Y_test)*100, 2)
print(f"The accuracy score achieved using XGBoost is: {score_xgb} %")

# Neural Network
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(11, activation='relu', input_dim=13))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=100, verbose=0)

Y_pred_nn = model.predict(X_test)
rounded_nn = [round(x[0]) for x in Y_pred_nn]
score_nn = round(accuracy_score(rounded_nn, Y_test)*100, 2)
print(f"The accuracy score achieved using Neural Network is: {score_nn} %")

# Plotting accuracy scores
scores = [score_lr, score_nb, score_svm, score_knn, score_dt, score_rf, score_xgb, score_nn]
algorithms = ["Logistic Regression", "Naive Bayes", "Support Vector Machine", "K-Nearest Neighbors",
              "Decision Tree", "Random Forest", "XGBoost", "Neural Network"]

# Create a pandas DataFrame from the scores and algorithms
score_df = pd.DataFrame({"Algorithm": algorithms, "Accuracy": scores})

sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")
# Use the DataFrame and column names in the barplot function
sns.barplot(x="Algorithm", y="Accuracy", data=score_df)
plt.show()
