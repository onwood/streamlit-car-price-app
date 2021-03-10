import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def run_home_app():
    st.title('차량 구매액을 예측하는 앱')
    st.write('왼쪽의 사이드바에서 선택하세요')
    st.image('https://img1.daumcdn.net/thumb/R720x0/?fname=http%3A%2F%2Ft1.daumcdn.net%2Fliveboard%2Fchutcha%2Fac86653be843413a8ac15e6f53938582.JPG')