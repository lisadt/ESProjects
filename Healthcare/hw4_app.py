# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 13:48:18 2020

@author: escag
"""


import streamlit as st
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import seaborn as sns
import xgboost as xgb
from matplotlib.pyplot import style
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from pdpbox import pdp
import random
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection  import RepeatedStratifiedKFold

dict1b={'case_id': np.int32, 'Hospital_code': np.int8, 'City_Code_Hospital': np.int8,
       'Available Extra Rooms in Hospital': np.int8, 'Bed Grade': np.float32, 'patientid': np.int32,
       'City_Code_Patient': np.float32, 'Visitors with Patient': np.int8, 'Admission_Deposit': np.float32,
      }
style.use('ggplot')
df = pd.read_csv("https://raw.githubusercontent.com/lisadt/es_repo280720/master/train_data.csv",skiprows=lambda x: x > 0 and random.random() > 0.1, dtype=dict1b)

st.title("Unit 4 - Homework- Healthcare Analytics II")
st.write(df)


df.Stay.replace('More than 100 Days', '>100', inplace=True)

section = st.sidebar.radio('Choose Application Sectio',
                           ['Data Explorer', 'Model Explorer'])

if section == 'Data Explorer':
    st.write(df)
    x_axis=st.sidebar.selectbox('Enter the column for the X-Axis', df.columns.tolist())
    y_axis=st.sidebar.selectbox('Enter Column for the Y-Axis', df.select_dtypes(include=np.number).columns.tolist(), index=1)
    
    x_axis_cat=st.sidebar.selectbox('Enter the column for the X-Axis', df.select_dtypes(include='object').columns.tolist())
    chart_type = st.sidebar.selectbox("What type of chart do you want to have",
                                      ['Line', 'Bar','Catplot_Xonly'])
    st.subheader(f'Data Visualizer for: {x_axis}, {y_axis}')
    
    if chart_type == 'Line':
        st.line_chart(df.groupby(x_axis)[y_axis].mean())
    elif chart_type == 'Bar':
        st.bar_chart(df.groupby(x_axis)[y_axis].mean())
    elif chart_type== 'Catplot_Xonly':
        chart=sns.catplot( kind='count', x=x_axis_cat, data=df, aspect=1.5)
        st.pyplot(chart)
        
if section =='Model Explorer':

    st.cache()
    pipe = make_pipeline(OrdinalEncoder(use_cat_names=True), xgb.XGBClassifier())
    
    st.cache()
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['Stay', 'case_id', 'Stay_int'], axis=1), df['Stay'], test_size=0.2, stratify=y,random_state=65)    
    
#if page == 'Model Explorer':
    num_rounds      = st.sidebar.number_input('Number of Boosting Rounds',
                                 min_value=100, max_value=800, step=100)
    
    tree_depth      = st.sidebar.number_input('Tree Depth',
                                 min_value=3, max_value=9, step=1, value=3)
    
               
    pipe[1][1].set_params(objective='multi:softmax',learning_rate=0.1,n_estimators=num_rounds, max_depth=tree_depth)
    
    X_train, X_val, y_train, y_val = train_test_split(df.drop('Stay', axis=1), df['Stay'], test_size=validation_size, random_state=random_state, num_class=11, eval_metric='merror')
    
    pipe.fit(X_train, y_train)
    
    mod_results = pd.DataFrame({
            'Boosting Rounds': num_rounds,
            'Tree Depth': tree_depth,
            'Learning Rate': learning_rate,
            'Training Score': pipe.score(X_train, y_train),
            'Validation Score': pipe.score(X_val, y_val)
            }, index=['Values'])
    
    st.subheader("Model Results")
    st.table(mod_results)
    

    st.subheader("Real vs Predicted Validation Values")
   
        