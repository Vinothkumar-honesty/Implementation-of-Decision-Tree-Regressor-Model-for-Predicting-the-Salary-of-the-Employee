## Algorithm
1. Import required libraries for data handling, preprocessing, modeling, and evaluation.
2. Load the dataset from the CSV file into a pandas DataFrame.
3. Check for null values and inspect data structure using .info() and .isnull().sum().
4. Encode the categorical "Position" column using LabelEncoder.
5. Split features (Position, Level) and target (Salary), then divide into training and test sets.
6. Train a DecisionTreeRegressor model on the training data.
7. Predict on test data, calculate mean squared error and R² score, and make a sample prediction.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: VINOTHKUMAR R
RegisterNumber:  212224040361
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
```

```
# load the dataframe
data = pd.read_csv("/content/Salary.csv")
```
```
# display head values
data.head()
```
```
# display dataframe information
data.info()
```
```
# display the count of null values
data.isnull().sum()
```
```
# encode postion using label encoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
```
```
# defining x and y and splitting them
x = data[["Position", "Level"]]
y = data["Salary"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
# predicting test values with model
y_pred = dt.predict(x_test)
mse = metrics.mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error : {mse}")
```
```
r2 = metrics.r2_score(y_test, y_pred)
print(f"R Square : {r2}")
```
```
dt.predict(pd.DataFrame([[5,6]], columns=["Position", "Level"]))
```



## Output:



**Head Values**

![1](https://github.com/user-attachments/assets/5bffa929-1de6-4307-8e18-f16b3284735b)


**DataFrame Info**

![2](https://github.com/user-attachments/assets/02fd7b0f-af67-4322-b5cb-ff76f1f12fa6)


**Sum - Null Values**

![3](https://github.com/user-attachments/assets/746c3842-21d4-4106-ad5c-5ca1c09e48ce)


**Encoding position feature**

![4](https://github.com/user-attachments/assets/8737ed1b-451f-45b1-9ec5-6bb9ad1f9420)



**Mean Squared Error**

![5](https://github.com/user-attachments/assets/08f30b1b-84fc-4bb6-849e-989badcc25e5)


**R2 Score**

![6](https://github.com/user-attachments/assets/8c0c6d3b-12e0-4acb-b83f-0f40ec603d8a)


**Final Prediction on Untrained Data**

![7](https://github.com/user-attachments/assets/351dcea1-6732-4129-9100-7582ed8b3b55)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
