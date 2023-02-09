#Importing required modules
import pandas as pd
import pandas_profiling as pp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

#reading the train data and test data
train = pd.read_csv(r"heart_rate_prediction_train.csv")
test = pd.read_csv(r"heart_rate_prediction_test.csv")
train
test

x = train[['sampen']].values
y = train[['higuci']].values

m = test[['sampen']].values
n = test[['higuci']].values

x_train, x_test, y_train, y_test = x ,m, y, n

#build the regression model
linreg = LinearRegression()
linreg.fit(x_train, y_train)

#Prediction of Test set Result
test_pred = linreg.predict(x_test)
train_pred = linreg.predict(x_train)

#Creating Stress Graph
train['condition'].unique()
conditions = dict(train['condition'].value_counts())
labels = list(conditions.keys())
counts = list(conditions.values())
plt.bar(labels,counts, color ='green',width = 0.4)
plt.show()

#Projecting analysed data to output.html
le = preprocessing.LabelEncoder()
le.fit(train['condition'])
train['condition'] = le.transform(train['condition'])
test['condition'] = le.transform(test['condition'])
profile = pp.ProfileReport(train)
profile.to_file("output.html")

#Creating Heat Map 
plt.figure(figsize=(12,10))
corr = train.corr()
sns.heatmap(corr, annot=False, cmap=plt.cm.Reds)
plt.show()

#Visualizing the training set results
plt.scatter(x_train, y_train, color = "green", cmap='viridis')
plt.colorbar()
plt.plot(x_train, train_pred, color = "red")
plt.title("Sampen vs Higuci (Training set)")
plt.xlabel("Sampen")
plt.ylabel("Higuci")
plt.show()

#Visualizing the test set results
plt.scatter(x_test, y_test, color = "blue", cmap='blues')
plt.colorbar()
plt.plot(x_train, train_pred, color = "red")
plt.title("Sampen vs Higuci (Test set)")
plt.xlabel("Sampen")
plt.ylabel("Higuci")
plt.show()

#Printing Training and Test Data sets
print("Training set top 5 rows:")
print(train.head(5))
print("Training set last 5 rows:")
print(train.tail(5))
print("Test set top 5 rows:")
print(test.head(5))
print("Test set last 5 rows:")
print(test.tail(5))
