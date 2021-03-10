import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def run_eda_app():
    df = pd.read_csv('data/Car_Purchasing_Data.csv', encoding='ISO-8859-1')
    X = df.iloc[ : , 3:-2+1]
    
    st.title('사용된 데이터 확인')
    st.write('이 앱은 고객 데이터와 자동차 구매액에 대한 내용을 담고 있습니다. 특정 정보를 입력하면 차량을 구매할 예상액을 도출해낼 수 있는 앱입니다.')
    columns = X.columns

    st.write('\n')
    st.write('\n')

    multiselected_colums = st.multiselect('컬럼을 선택하세요',columns)
    
    if multiselected_colums != []:
        st.dataframe(df[multiselected_colums])    
        
        st.write('\n')
        st.write('\n')

        radio_menu = ['통계치','상관 관계']
        selected_radio = st.radio('선택한 데이터의 가공된 데이터를 확인해보세요', radio_menu)
        
        if selected_radio == '상관 관계':
            st.dataframe(df[multiselected_colums].corr())
        if selected_radio == '통계치':
            st.dataframe(df[multiselected_colums].describe())
