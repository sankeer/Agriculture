import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pickle
df = pd.read_csv('C:/Users/admin/PycharmProjects/Agriculture/Crop_recommendation.csv')
df.head()
df.shape#2200,8
df.isnull().sum()#no null values found
df.dtypes
X=df.drop("label",axis=1)
Y=df['label']
Z=Y
print(Z)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder=LabelEncoder()
Y=labelencoder.fit_transform(Y)
print(Y)
dictionary = {}

for key, value in zip(Y, Z):
    if key not in dictionary:
        dictionary[key] = value
        print(key,value)
print(dictionary)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
from sklearn.ensemble import RandomForestClassifier
classifier =  RandomForestClassifier()
classifier.fit(X_train,y_train)
#@title RANDOM FOREST
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test, y_pred)
print(cm)
acc=accuracy_score(y_test,y_pred)
print(acc)
pickle.dump(classifier, open('model.pkl', 'wb'))
model= pickle.load(open('model.pkl','rb'))
index=model.predict([[90,42,43,20,82,6.5,204]])
decoded_data = labelencoder.inverse_transform(Y)
print(decoded_data)
print(decoded_data[index])