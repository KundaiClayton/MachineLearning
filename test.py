import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style



data=pd.read_csv("student-mat.csv", sep=";")

data=data[["G1","G2","G3","studytime","failures","absences"]]
#print(data)
predict="G3"

X=np.array(data.drop([predict],1))

y=np.array(data[predict])
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.1)

##training model to get the highest score and store it in our pickle
"""
best=0
for _ in range(30):

    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.1)

    linear=linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc=linear.score(x_test, y_test)
    print(acc)

    if acc>best:
        best=acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

    print(best)
"""
pickle_in=open("studentmodel.pickle","rb")
linear =pickle.load(pickle_in)

print("Coefficient:\n", linear.coef_)
print("Intercept:\n",linear.intercept_)

predictions=linear.predict(x_test)
for i in range(len(predictions)):
    print(predictions[i],x_test[i],y_test[i])

p="failures"
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final grade")
pyplot.show()
