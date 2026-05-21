# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook


## Algorithm
1. Load and Preprocess Data: Read the dataset, drop unnecessary columns, and convert categorical variables into numerical codes using .astype('category') and .cat.codes.

2. Define Variables: Split the dataset into features (X) and target variable (Y), and initialize a random parameter vector theta.

3. Implement Functions: Define the sigmoid, loss, gradient_descent, and predict functions for logistic regression.

4. Train Model: Use gradient descent to optimize the parameters theta over a specified number of iterations.

5. Evaluate and Predict: Calculate accuracy of predictions on the training data, and demonstrate predictions with new sample data.
## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: RICKY DHARMESH P
RegisterNumber:  25016025  
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("\nAccuracy Score:\n",accuracy)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix:\n",confusion)

from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print("\nClassification Report:\n",cr)
```

## Output:

<img width="657" height="352" alt="image" src="https://github.com/user-attachments/assets/d3be513c-433c-4dc5-80a1-8d28efc3bd7c" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

