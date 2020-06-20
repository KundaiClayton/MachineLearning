import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style


data=pd.read_csv("student-por.csv", sep=";")
#print(data.head())
"""filter the data you wanna play with"""
data=data[["age","famrel","traveltime","studytime","failures","freetime","goout","Dalc","Walc","health","absences","G1","G2","G3"]]

predict="G3"

W=np.array(data.drop([predict],1)) #features
y=np.array(data[predict]) # labels

"""split data into testing and training adata"""
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(W,y,test_size=0.1)
best=0

best=0
for _ in range(100):
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(W,y,test_size=0.1)
    linear=linear_model.LinearRegression()
    linear.fit(x_train,y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy>best:
        best=accuracy
        with open("portuguesegrades.pickle","wb") as f:
            pickle.dump(linear,f)

#load model
pickle_in=open("portuguesegrades.pickle","rb")
linear=pickle.load(pickle_in)


print("*********************************************")
print(linear.coef_)
print(linear.intercept_)
print("*********************************************")

predictions=linear.predict(x_test)
for i in range(len(predictions)):
    print(predictions[i],x_test[i],y_test[i])

"""Plot model"""
plot="absences"
plt.scatter(data[plot],data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Portuguese Grade")

plt.show()
