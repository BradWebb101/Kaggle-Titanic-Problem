# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:27:23 2019

@author: bradw
"""

# data analysis and wrangling
import pandas as pd

#Timing functions 
import time

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#Timing start
start_time = time.time()

#Loading data frames into pandas
train_df = pd.read_csv('train.csv',index_col='PassengerId')
test_df = pd.read_csv('test.csv',index_col='PassengerId')

#Checking basic data details
train_df.info()
print('-'*40)
test_df.info()

#Understanding data types of files
train_describe = train_df.describe()
print('-'*40)
test_describe = test_df.describe()

#Understanding the make up of the objects
print(train_df.describe(include=['O']))

#Checking the correlation of data to survived
corr = train_df.corr()
sns.heatmap(corr)
plt.show()

#Checking which rows have missing data 
print('data missing from dataset')
print(pd.isna(train_df).sum())


#Dropping datasets that have too many missing and are not useful
train_df.drop(['Ticket','Cabin'],inplace=True,axis=1)
test_df.drop(['Ticket','Cabin'],inplace=True,axis=1)

#Checking survival rates on PClass and Gender
print(train_df[['Pclass','Survived']].groupby('Pclass').mean().sort_values(by='Survived',ascending=False))
print(train_df[['Sex','Survived']].groupby('Sex').mean().sort_values(by='Survived',ascending=False))
print(train_df[['SibSp','Survived']].groupby('SibSp').mean().sort_values(by='Survived',ascending=False))
print(train_df[['Parch','Survived']].groupby('Parch').mean().sort_values(by='Survived',ascending=False))
print(train_df[['Embarked','Survived']].groupby('Embarked').mean().sort_values(by='Survived',ascending=False))
print(train_df[['Fare','Survived']].groupby('Survived').mean().sort_values(by='Survived',ascending=False))

#Concat data into single frame for data cleaning and imputation
df = pd.concat([train_df,test_df])

#Creating plots to see the spread of age data
print('Age Distribution Box Plot')
sns.boxplot(df['Age'],orient='v')
plt.show()
print('Histogram of Age Distribution')
sns.distplot(df['Age'].dropna())
plt.show()   

#Checking fare cost affecting survived
fig,axis = plt.subplots()
axis.boxplot([train_df['Fare'][train_df['Survived'] == 0],train_df['Fare'][train_df['Survived'] == 1]],labels=('DNS','Survived'))
axis.set_ylabel('Fare')
axis.set_title('Fare distribution on passengers survival')
plt.show()



#Converting data into numerical forms and extracting other key variables
#Functions for data cleaning and imputing values in Nan
def data_wrangler(dataframe):
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Mlle','Countess','Mme','Lady','Dona'], 'Mrs')
    df['Title'] = df['Title'].replace(['Ms'], 'Miss')
    df['Title'] = df['Title'].replace(['Major','Sir','Capt','Col','Don','Rev'], 'Mr')
    df['Title'] = df['Title'].replace(['Jonkheer','Master'], 'Master')
    df['Title'].loc[797] = 'Dr. F'
    dataframe['Sex No.'] = dataframe.Sex.map({'female':0,'male':1})
    dataframe['Embarked'] = dataframe['Embarked'].fillna('S')
    dataframe['Embarked No.'] = dataframe.Embarked.map({'S':0,'Q':1,'C':2})
    dataframe['Family Size'] = dataframe['SibSp'] + dataframe['Parch']
    dataframe['Title No.'] = dataframe.Title.map({'Mr':0, 'Mrs':1,'Master':2, 'Miss':3, 'Dr':4, 'Dr. F':5})
    dataframe = dataframe.drop(['Name','SibSp','Parch','Embarked'],axis=1,inplace=True)
    
data_wrangler(df)

#Binning ages into categories and using names as a varaible to add in details
pd.crosstab(df['Title'], df['Sex'])
print(df.groupby('Title').count())

#Age table 
age_table_median = df[['Title','Age','Sex','Pclass']].groupby(['Sex','Pclass','Title']).median()
age_table_count = df[['Title','Age','Sex','Pclass']].groupby(['Sex','Pclass','Title']).count()
#No of doctors is a bit low for significant age impute. 
#Checking doctor na on age to make sure no miss impute of data due to small sample 
print(df[['Title','Age']].groupby('Title').count())
#1 doctor Nan and is male so sample size ok for age impute


#Age impute for Nan on PClass and Title variables
def title_age_impute():
    age_table = df[['Title','Age','Sex','Pclass']].groupby(['Sex','Pclass','Title']).median()
    for i in range(len(df)):
        if pd.isna(df['Age'].iloc[i]) == True:
            df['Age'].iloc[i] = int(age_table.loc[(str(df['Sex'].iloc[i]),int(df['Pclass'].iloc[i]),str(df['Title'].iloc[i]))])
       
title_age_impute()

#Dropping non numerical columns from DataFrame
df = df.drop(['Sex','Title'],axis=1)

#Splitting data into train and test data after data wrangling
train_df = df[pd.notna(df['Survived'])]
test_df = df[pd.isna(df['Survived'])]
test_df = test_df.drop(['Survived','Fare'],axis=1)


#Creating Train test split to test logistic regression model
X_train, X_test, y_train, y_test = train_test_split(train_df.drop(['Survived','Fare'],axis=1),train_df['Survived'])

#Logistic regression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
log_predict = logmodel.predict(X_test)

#Testing effectiveness of the model
print(confusion_matrix(y_test,log_predict))
print(classification_report(y_test,log_predict))

#Knn test no scaler
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knn_predict = knn.predict(X_test)

#Testing sucess of the model KNN
print(confusion_matrix(y_test,knn_predict))
print(classification_report(y_test,knn_predict))


#Random Forst classifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_predict = rfc.predict(X_test)

#testing model on test data
print(confusion_matrix(y_test,rfc_predict))
print(classification_report(y_test,rfc_predict))

#Ada Boost classifier
ada = AdaBoostClassifier()
ada.fit(X_train,y_train)
ada_predict = ada.predict(X_test)

#testing model on test data
print(confusion_matrix(y_test,ada_predict))
print(classification_report(y_test,ada_predict))

#Decision Tree
decT = DecisionTreeClassifier()
decT.fit(X_train,y_train)
decT_predict = decT.predict(X_test)

#testing model on test data
print(confusion_matrix(y_test,decT_predict))
print(classification_report(y_test,decT_predict))

#Predictions out
submission = pd.DataFrame(index=test_df.index,columns=['Survived'])
submission['Survived'] = rfc.predict(test_df)
submission = submission.astype(int)
submission.info()

submission.to_csv('Submission.csv')


#Time end
print("Code ran in %s seconds" % (time.time() - start_time))
