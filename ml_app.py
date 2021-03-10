import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
import pickle
from sklearn.preprocessing import MinMaxScaler


def run_ml_app():
    st.title('차량 구매액 예측')
    st.write('성별(Gender), 연령(Age), 연봉(Annual Salary), 카드빚(Credit Card Debt), 순자산(Net Worth)을 직접 입력해서 차량 구매액을 예측해보세요.')
    
    df = pd.read_csv('data/Car_Purchasing_Data.csv', encoding='ISO-8859-1')
    X = df.iloc[ : , 3:-2+1]
    y = df['Car Purchase Amount']
    
    model = tf.keras.models.load_model('data/by-type-ANN-onwood-original.h5')
    sc_X = joblib.load('data/sc_X.pkl')
    sc_y = joblib.load('data/sc_y.pkl')
    
    # gender 선택 / age /  annual salary / credit card debt / net worth
    
    gender = st.radio('Gender',['Female','Male'])
    if gender == 'Female':
        gender = 0
    else:
        gender = 1        

    age = st.number_input('Age', 1)
    salary = st.number_input('Annual Salary', 0.0)
    debt = st.number_input('Credit Card Debt', 0.0)
    net_worth = st.number_input('Net Worth', 0.0)

    new_data = np.array([gender, age, salary, debt, net_worth]).reshape(1,-1)  
    
    new_data = sc_X.transform(new_data)
    new_data_pred = model.predict(new_data)

    predicted_data = sc_y.inverse_transform(new_data_pred)
    st.write('예측 값은 {}입니다.'.format(predicted_data))