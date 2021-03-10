import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def main():
    df = pd.read_csv('data/Car_Purchasing_Data.csv', encoding='ISO-8859-1')

    menu = ['Home', 'Data', 'Predict', 'About']
    choice = st.sidebar.selectbox('Menu', menu)
    X = df.iloc[ : , 3:-2+1]
    y = df['Car Purchase Amount']    

    if choice == 'Home':
        st.title('차량 구매액을 예측하는 앱')
        st.image('https://img1.daumcdn.net/thumb/R720x0/?fname=http%3A%2F%2Ft1.daumcdn.net%2Fliveboard%2Fchutcha%2Fac86653be843413a8ac15e6f53938582.JPG')
    
    if choice == 'Data':
        st.title('사용된 데이터 확인')
        columns = X.columns
        multiselected_colums = st.multiselect('컬럼을 선택하세요',columns) 
        if multiselected_colums != []:
            st.dataframe(df[multiselected_colums])
            if st.button('상관 관계'):
                st.dataframe(df[multiselected_colums].corr())

    if choice == 'Predict':
        st.title('차량 구매액 예측')
        model = tf.keras.models.load_model('data/by-type-ANN-onwood-original.h5')
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
        



        mms_X = MinMaxScaler()
        X = mms_X.fit_transform(X)
        
        mms_y = MinMaxScaler()
        y = y.values.reshape(-1,1)
        y = mms_y.fit_transform(y)     
        
        new_data = mms_X.transform(new_data)
        new_data_pred = model.predict(new_data)

        predicted_data = mms_y.inverse_transform(new_data_pred)
        st.write('예측 값은 {}입니다.'.format(predicted_data))


if __name__ == '__main__':
    main()