# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 20:51:18 2019

@author: bradw
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

train = pd.read_csv('train.csv',index_col='PassengerId')
test = pd.read_csv('test.csv')

train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()

#Dodgy removal of Nan from Test['Fare']

test.fillna(value=0)

#Adding in age as mean for class type as function
def pred_age(dataframe):
    x = 0
    y = 0
    nan = 0
    sum1 = 0
    sum2 = 0
    sum3 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    while x < len(dataframe['Age']):
        if np.isnan(dataframe['Age'].iloc[x]) == True:
            #print('Nan Found')
            nan+=1
            x+=1
            
        else:
            if dataframe['Pclass'].iloc[x] == 1:
                sum1 = sum1 + dataframe['Age'].iloc[x]
                count1 = count1 + 1
                #print('Class 1 Found')
                x+=1
               
                
            elif dataframe['Pclass'].iloc[x] == 2:
                sum2 = sum2 + dataframe['Age'].iloc[x]
                count2 = count2 + 1
                #print('Class 2 Found')
                x+=1
                
            else:
                count3 = count3 + 1
                #print('Class 3 Found')
                x+=1   
                
    while y < len(dataframe['Age']):
        if np.isnan(dataframe['Age'].iloc[y]) == False:
           # print('Not a Nan')
            y+=1
        else:
            if dataframe['Pclass'].iloc[y] == 1:
                dataframe['Age'].iloc[y] = (sum1/count1)
                #print('Class 1 NAN, filled')
                y+=1
            elif dataframe['Pclass'].iloc[y] == 2:
                dataframe['Age'].iloc[y] = (sum2/count2)
                #print('Class 2 NAN, filled')
                y+=1
            else:
                dataframe['Age'].iloc[y] = (sum3/count3)
                #print('Class 3 NAN, filled')
                y+=1
    
    print('Nans found ' + str(nan))
    sns.heatmap(dataframe.isnull(),yticklabels=False,cbar=False,cmap='viridis')
    plt.show()
    return

#Simulate missing age variables for test and train set    
pred_age(train)
pred_age(test)     

 

#Dropping non essential data from DF for Logical Regression
def dummy_var(dataframe):
    dataframe['dummy_sex'] = pd.get_dummies(dataframe['Sex'],drop_first=True)
    dataframe[['dummy_port1','dummy_port2']] = pd.get_dummies(dataframe['Embarked'],drop_first=True)
    dataframe.drop(['Name','Sex','Ticket','Embarked'],axis=1,inplace=True)
    
dummy_var(train)
dummy_var(test)

#Training Model and adding variables
X_train = train.drop('Survived',axis=1)
y_train = pd.DataFrame(train['Survived'])
X_test = test.drop('PassengerId',axis=1).fillna(value=0)
y_test = pd.DataFrame(test['PassengerId'])

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

y_test['Survived'] = predictions

y_test.to_csv('Model_preictions.csv',index=False)
