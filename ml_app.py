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
    

    
    # gender 선택 / age /  annual salary / credit card debt / net worth
    
    gender = st.radio('성별(Gender)',['Female','Male'])
    if gender == 'Female':
        gender = 0
    else:
        gender = 1        

    age = st.number_input('연령(Age)', 0)
    salary = st.number_input('연봉(Annual Salary)', 0.0)
    debt = st.number_input('카드빚(Credit Card Debt)', 0.0)
    net_worth = st.number_input('순자산(Net Worth)', 0.0)

    input_data_list=[gender, age, salary, debt, net_worth]

    model = tf.keras.models.load_model('data/by-type-ANN-onwood-original.h5')
    sc_X = joblib.load('data/sc_X.pkl')
    sc_y = joblib.load('data/sc_y.pkl')

    new_data = np.array(input_data_list).reshape(1,-1)  
    
    new_data = sc_X.transform(new_data)
    new_data_pred = model.predict(new_data)
    predicted_data_final = sc_y.inverse_transform(new_data_pred)
    if st.button('결과'):
        st.write('예측 값은 {:,.2f}$입니다.'.format(predicted_data_final[0,0]))