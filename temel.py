import pandas as p
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
data = p.read_csv("insurance.csv")



data["sex"] = data["sex"].astype("category")
data["smoker"] = data["smoker"].astype("category")
data["region"] = data["region"].astype("category")

#print(data.dtypes)

#print(data.describe().T)
smoke_data = data.groupby("smoker").mean().round(2)
#print(smoke_data)
#print(sns.set_style("whitegrid"))

sns.pairplot(data[["age","bmi","charges","smoker"]],
             hue="smoker",
             palette="Set1",
             height= 3)

sns.heatmap(data.corr(), annot=True)

data = pd.get_dummies(data)
#print(data.columns)
y = data["charges"]
X = data.drop("charges",axis = 1)

#print(data.columns)

#plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.80,random_state=1)
lr = LinearRegression()
lr.fit(X_train,y_train)
print(lr.score(X_test,y_test).round(3))

print(lr.score(X_train,y_train).round(3))


data_new = X_train[:]

liste = lr.predict(data_new)

print("Toplam doalr", sum(liste))
print(lr.predict(data_new))

liste2 = y_train[:]

print("Greçek toplam", sum(liste2))


print("Aradaki dolar farkı", sum(liste)-sum(liste2))

