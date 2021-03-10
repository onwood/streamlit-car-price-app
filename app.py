import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from ml_app import run_ml_app
from eda_app import run_eda_app
from home_app import run_home_app

def main():
    

    menu = ['Home', 'EDA', 'ML', 'About']
    choice = st.sidebar.selectbox('Menu', menu)
    

    if choice == 'Home':
        run_home_app()

    if choice == 'EDA':
        run_eda_app()

    if choice == 'ML':
        run_ml_app()


if __name__ == '__main__':
    main()