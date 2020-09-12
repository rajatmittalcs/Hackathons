import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from datetime import datetime as dt
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier


#Importing Train test and submission file

train=pd.read_csv('Train_gQ1XM0h.csv')

test=pd.read_csv('Test_wyCirpO_aS3XPkc.csv')

submission=pd.read_csv('sample_submission_V7OCXsi.csv')

#Making collective data of train n test to do data cleaning on both together
alldata=pd.concat([train,test],axis=0).reset_index(drop=True)

df=alldata.copy()

#Highly correlated with Manager_Business & Manager_Num_Products 
df=df.drop(['Manager_Business2','Manager_Num_Products2'],axis=1)

df=df.drop(['Applicant_City_PIN'],axis=1)

#Converting all dates to proper format to extract information
df['Application_Receipt_Date']=pd.to_datetime(df['Application_Receipt_Date'],format='%m/%d/%Y')

df['Manager_DOJ']=pd.to_datetime(df['Manager_DOJ'],format='%m/%d/%Y')

df['Applicant_BirthDate']=pd.to_datetime(df['Applicant_BirthDate'],format='%m/%d/%Y')

df['Manager_DoB']=pd.to_datetime(df['Manager_DoB'],format='%m/%d/%Y')

#Assigning Confirmation 2 and Probabtion 1 as they are ordinal
df['Manager_Status']=df['Manager_Status'].replace(['Probation','Confirmation'],[1,2])

#All new columns extracted below are in respect to Application Date and in number of days so that we know the statistics at the time of application
df['Manager_Experience_at_appl']=(df['Application_Receipt_Date'] - df['Manager_DOJ']).dt.days

df['Mgr_Age_at_appl']=(df['Application_Receipt_Date'] - df['Manager_DoB']).dt.days

df['Agent_age_at_appl']=(df['Application_Receipt_Date'] - df['Applicant_BirthDate']).dt.days

# Handling Null values with mean values
df['Manager_Experience_at_appl'].fillna(int(df['Manager_Experience_at_appl'].mean()),inplace=True)
df['Mgr_Age_at_appl'].fillna(int(df['Mgr_Age_at_appl'].mean()),inplace=True)
df['Agent_age_at_appl'].fillna(int(df['Agent_age_at_appl'].mean()),inplace=True)

#Dropping dates column as info extracted
df.drop(['Manager_DoB','Applicant_BirthDate','Manager_DOJ','Application_Receipt_Date'],axis=1,inplace=True)

#Assigning values acc to qualification level and imputing null values with 0. And manager designation acc to their levels
df['Applicant_Qualification']=df['Applicant_Qualification'].replace(['Class X','Class XII','Graduate','Masters of Business Administration','Associate/Fellow of Institute of Company Secretories of India','Associate / Fellow of Institute of Chartered Accountans of India','Certified Associateship of Indian Institute of Bankers','Associate/Fellow of Insurance Institute of India','Associate/Fellow of Institute of Institute of Costs and Works Accountants of India','Associate/Fellow of Acturial Society of India','Professional Qualification in Marketing','Others'],[1,2,3,4,4,4,4,4,4,4,4,5])

df['Applicant_Qualification'].fillna(0,inplace=True)

df['Applicant_Qualification']=df['Applicant_Qualification'].astype(int)

df['Manager_Joining_Designation']=df['Manager_Joining_Designation'].replace(['Other','Level 1', 'Level 2', 'Level 3',  'Level 4', 'Level 5','Level 6','Level 7'],[0,1,2,3,4,5,6,7])

df['Manager_Current_Designation']=df['Manager_Current_Designation'].replace(['Other','Level 1', 'Level 2', 'Level 3',  'Level 4', 'Level 5','Level 6','Level 7'],[0,1,2,3,4,5,6,7])

#df=pd.get_dummies(df, columns=['Applicant_Marital_Status','Applicant_Occupation'], prefix=['Apl_Marital_Stat',''],dtype=int )

# Handling remaining categorical data by factorizing.
# Tried with One hot encoding above but accuracy suffers a lot in that case.
df['Applicant_Marital_Status']=pd.factorize(df['Applicant_Marital_Status'])[0]
df['Applicant_Occupation']=pd.factorize(df['Applicant_Occupation'])[0]
df['Manager_Gender']=pd.factorize(df['Manager_Gender'])[0]
df['Applicant_Gender']=pd.factorize(df['Applicant_Gender'])[0]
df['Applicant_Gender']=df['Applicant_Gender'].replace(0,2)
df['Manager_Gender']=df['Manager_Gender'].replace(0,2)
df['Applicant_Marital_Status']=df['Applicant_Marital_Status'].replace(0,4)


#df=pd.get_dummies(df, columns=["Applicant_Gender",'Manager_Gender','Applicant_Marital_Status','Applicant_Occupation'], prefix=["App_gender",'Mgr_gender','Apl_Marital_Stat',''],dtype=int )

#Dividing Training and Test Data and handling for making it algo ready
xtrain=df.iloc[:9527,:]

xtest=df.iloc[9527:,:]

#Null imputations and penalization acc to the feature 
xtest.fillna( {'Manager_Joining_Designation':-1,'Manager_Current_Designation':-1,'Manager_Grade':0,'Manager_Status':0,'Manager_Num_Application':0,'Manager_Num_Coded':-1,'Manager_Business':0,'Manager_Num_Products':0}, inplace=True )

xtrain =xtrain.dropna(subset=['Manager_Status'])

X_train=xtrain.drop(['ID','Business_Sourced'],axis=1)

y_train=xtrain.Business_Sourced

X_test=xtest.drop(['ID','Business_Sourced'],axis=1)
test_ids=xtest.ID

##X_train, X_test, y_train, y_test = tt(xtrain, ytrain, test_size=0.20, random_state=10)

#Scaling the test and train data to converge faster and better
X_train=scale(X_train)
poly = PolynomialFeatures(3)
X_train=poly.fit_transform(X_train)

X_test=scale(X_test)
poly = PolynomialFeatures(3)
X_test=poly.fit_transform(X_test)

#########################################################################################

#Training using Multi Layer Perceptron (ANN) ML algo from sklearn. Checking F1 score,accuracy and ROC-AUC
mlp = MLPClassifier(hidden_layer_sizes=(20, 3), max_iter=150, alpha=1e-4,solver='sgd', verbose=10, tol=1e-4, random_state=1,learning_rate_init=.1)

mlp.fit(X_train,y_train)

y_pred = mlp.predict(X_train)
y_pred=pd.Series(y_pred)
f1_score(y_train,y_pred)
accuracy_score(y_train,y_pred)
roc_auc_score(y_train,y_pred)

#Predicting on Test Data and sending to a csv file for submission

test_pred=mlp.predict(X_test)
pd.DataFrame({'ID':test_ids,'Business_Sourced':test_pred}).to_csv('rajat_submission.csv',index=False)

################################################################################
