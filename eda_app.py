import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def run_eda_app():
    df = pd.read_csv('data/Car_Purchasing_Data.csv', encoding='ISO-8859-1')
    # X = df.iloc[ : , 3:-2+1]
    
    st.title('사용된 데이터 확인')
    st.write('이 앱은 고객 데이터와 자동차 구매액에 대한 내용을 담고 있습니다. 특정 정보를 입력하면 차량을 구매할 예상액을 도출해낼 수 있는 앱입니다.')
    columns = df.columns
    # print(columns)
    # print(df.dtypes != object)

    st.write('\n')
    st.write('\n')

    number_columns = df.columns[df.dtypes != object]
    multiselected_colums = st.multiselect('컬럼을 선택하세요',number_columns)
    # st.write((df.loc(df['Age'] == df['Age'].max()).to_frame(),))

    if multiselected_colums != []:
        st.dataframe(df[multiselected_colums])    
        
        st.write('\n')
        st.write('\n')

        radio_menu = ['통계치','상관 관계']
        selected_radio = st.radio('선택한 데이터의 가공된 데이터를 확인해보세요', radio_menu)
        
        if selected_radio == '통계치':
            st.dataframe(df[multiselected_colums].describe())

        if selected_radio == '상관 관계':
            if len(multiselected_colums) > 1:
                st.dataframe(df[multiselected_colums].corr())
                
                fig = sns.pairplot(data = df[multiselected_colums])
                st.pyplot(fig)
            
            else:
                st.write('두 가지 이상의 데이터를 선택해주세요')
    
    else:
        st.write('선택한 컬럼이 없습니다.')

    # 컬럼 하나 선택하면 해당 컬럼의 min과 max에 해당하는 사람의 데이터 출력
    st.write('\n')
    st.write('\n')
    st.write('\n')

    st.subheader('원하는 컬럼의 최소값과 최대값 정보를 확인해보세요')
    number_column = df.columns[df.dtypes != object]    
    selected_col = st.selectbox('컬럼을 선택하세요', number_column)
    
    st.write('선택한 컬럼의 최소값 정보')
    min_data = df.loc[df[selected_col] == df[selected_col].min(),]
    st.dataframe(min_data)    

    st.write('선택한 컬럼의 최대값 정보')
    max_data = df.loc[df[selected_col] == df[selected_col].max(),]
    st.dataframe(max_data)  


    st.write('\n')
    st.write('\n')
    st.write('\n')

    st.subheader('고객의 이름으로 정보를 확인 해보세요')


    # 고객의 이름을 검색할 수 있는 기능 
    customer_name = st.text_input('검색하려는 고객의 이름을 입력하세요')
    name_filter = df.loc[df['Customer Name'].str.contains(customer_name, case = False) == True, ]
    st.dataframe(name_filter)

