# imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle

def create_model(data):
    x=data.drop("diagnosis",axis=1)
    y=data["diagnosis"] 
    ss=StandardScaler()
    x=ss.fit_transform(x)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
    model=LogisticRegression()
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print("Accuracy :",accuracy_score(y_true=y_test,y_pred=y_pred))
    print("Classification report :",classification_report(y_pred=y_pred,y_true=y_test))
    return model,ss

def get_data():
    data=pd.read_csv("./data.csv")
    # print(data.head())
    data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
    data["diagnosis"]=data["diagnosis"].map({"M":1,"B":0})
    return data

def main():
    # data
    data=get_data()
    
    # model
    model,ss=create_model(data)
    with open("model.pkl","wb") as f:
        pickle.dump(model,f)
    
    with open("scaler.pkl","wb") as g:
        pickle.dump(ss,g)
    
    
    
if __name__=="__main__":
    main()