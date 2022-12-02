import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier


import streamlit as st 
  


import joblib
import pickle


import datetime
 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.preprocessing import StandardScaler

st.title("Default Credit Card Clients") 

Knn=KNeighborsClassifier()    

    
df = pd.read_excel('b_file.xls')
df
limit_Bal = st.number_input("Enter your limit_Bal") 
status = st.radio("Select Gender: ", ('Male', 'Female')) 
  
if (status == 'Male'): 
    male=1
    female=0
else: 
    male=0
    female=1
    
    
status2 = st.radio("Select Marriage: ", ('OTHERS', 'married', 'single', 'divorce'))
  
if (status2 == 'OTHERS'): 
    
    OTHERS=1
    married=0
    single=0
    divorce=0
    
elif (status2 == 'married'): 
    
    OTHERS=0
    married=1
    single=0
    divorce=0
    
elif (status2 == 'single'): 
    
    OTHERS=0
    married=0
    single=1
    divorce=0
    
else:
    OTHERS=0
    married=0
    single=0
    divorce=1

    
    
status1 = st.radio("Select Education: ", ('grad', 'university', 'high school', 'other'))
  
if (status1 == 'university'): 
    
    university=1
    grad=0
    high_school=0
    other=0
    
elif (status1 == 'grad'):
    university=0
    grad=1
    high_school=0
    other=0
    
elif (status1=='high school'):
    university=0
    grad=0
    high_school=1
    other=0
    
else:
    university=0
    grad=0
    high_school=0
    other=1
   

    
Age = st.number_input("Enter your Age") 


PAY_1 = st.number_input("Enter your PAY_1") 

PAY_2 = st.number_input("Enter your PAY_2") 

PAY_3 = st.number_input("Enter your PAY_3") 

PAY_4 = st.number_input("Enter your PAY_4") 
PAY_5 = st.number_input("Enter your PAY_5") 
PAY_6 = st.number_input("Enter your PAY_6") 

BILL_AMT1= st.number_input("Enter your BILL_AMT1") 
BILL_AMT2= st.number_input("Enter your BILL_AMT2") 
BILL_AMT3= st.number_input("Enter your BILL_AMT3") 
BILL_AMT4= st.number_input("Enter your BILL_AMT4") 
BILL_AMT5= st.number_input("Enter your BILL_AMT5") 
BILL_AMT6= st.number_input("Enter your BILL_AMT6") 


PAY_AMT1= st.number_input("Enter your PAY_AMT1") 
PAY_AMT2= st.number_input("Enter your PAY_AMT2") 
PAY_AMT3= st.number_input("Enter your PAY_AMT3") 
PAY_AMT4= st.number_input("Enter your PAY_AMT4") 
PAY_AMT5= st.number_input("Enter your PAY_AMT5") 
PAY_AMT6= st.number_input("Enter your PAY_AMT6") 


if(st.button('Prédire le client')):
	
	x = df.drop('def_pay',axis=1)
	y = df['def_pay'] 
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
	knn = KNeighborsClassifier(23)
	knn_model = knn.fit(X_train, y_train)
	
	X_test=X_test.append({'AGE':Age,'BILL_AMT1':BILL_AMT1,'BILL_AMT2':BILL_AMT2,
                         'BILL_AMT3':BILL_AMT3,'BILL_AMT4':BILL_AMT4,'BILL_AMT5':BILL_AMT5,'BILL_AMT6':BILL_AMT6,
                         'PAY_AMT1':PAY_AMT1,'PAY_AMT2':PAY_AMT2,'PAY_AMT3':PAY_AMT3,
                         'PAY_AMT4':PAY_AMT4,'PAY_AMT5':PAY_AMT5,'PAY_AMT6':PAY_AMT6,'male':male,'female':female,'graduate school':grad,
                         'university':university,'PAY_1':PAY_1,'PAY_2':PAY_2,'PAY_3':PAY_3,'high school':high_school,'others':other,'OTHERS':OTHERS,'married':married,'single':single,'divorce':divorce,
                         'PAY_4':PAY_4,'PAY_5':PAY_5,'PAY_6':PAY_6,'LIMIT_BAL':limit_Bal}, ignore_index=True)
	y_pred_knn = knn_model.predict(X_test)
	
	if(y_pred_knn[-1]):
		st.text("Le client est risqué")
	else:
		st.text("Le client non risqué")
