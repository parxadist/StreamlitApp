import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

page_choice = st.sidebar.selectbox('Page Choice',['HomePage','EDA','Modeling'], )
selected_dataset = st.selectbox("Select dataset you want",['Loan Dataset', 'Water Dataset'])
if page_choice == 'HomePage':
    st.header('The poor attempt of recreating prediction site')
    st.image('image.png', width=700)
if page_choice == 'EDA':
    st.header('EDA Page')
    match selected_dataset:
        case 'Loan Dataset':
            data = pd.read_csv('loan_pred.csv')
            st.subheader('Initial dataset')
            st.dataframe(data)
            desc_df = round(data.describe()).T
            st.dataframe(desc_df)
            st.header('Visualisation')
            st.subheader('Imbalance')

            fig_countplot, ax_countplot = plt.subplots(figsize=(8, 5))
            sns.countplot(data=data, x='Loan_Status', ax=ax_countplot, palette='mako')
            ax_countplot.set_title('Countplot')
            st.pyplot(fig_countplot)

            fig_boxplot, ax_boxplot = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=data, ax=ax_boxplot, palette='mako')
            ax_boxplot.set_title('Boxplot')
            st.pyplot(fig_boxplot)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader('Null amounts')
                null_info = pd.DataFrame(data.isnull().sum(), columns = ['null'])
                st.dataframe(null_info )

            with col2:
                st.subheader('Imputation')
                numerical_strategy = st.radio('Imputation for numerical', ['mean','most_frequent'])
                categorical_strategy = st.radio('Imputation for categorical', ['most_frequent'])
                st.subheader('Feature engineering')
                numerical_cols = data.select_dtypes(exclude = 'object').columns
                categorical_cols = data.select_dtypes(include = 'object').columns
                to_undersample = st.checkbox('UnderSampling(not implemented)')
                to_clean_outlier = st.checkbox('Clean outlier(not implemented)')
                data['Credit_History'].fillna(1, inplace = True)
                if st.button('Run Data Preprocessing'):
                    preprocessor = ColumnTransformer([
                        ('cat', SimpleImputer(strategy= categorical_strategy), categorical_cols),
                        ('num', SimpleImputer(strategy= numerical_strategy), numerical_cols)
                    ],
                    remainder='passthrough')
                    data = pd.DataFrame(preprocessor.fit_transform(data), columns = categorical_cols.append(numerical_cols))
                    data['LoanAmount'] = data['LoanAmount'].apply(round)

        case 'Water Dataset':
            data = pd.read_csv('water_potability.csv')
            st.subheader('Initial dataset')
            st.dataframe(data)
            desc_df = round(data.describe()).T
            st.dataframe(desc_df)
            st.header('Visualisation')
            st.subheader('Imbalance')

            fig_countplot, ax_countplot = plt.subplots(figsize=(8, 5))
            sns.countplot(data=data, x='Potability', ax=ax_countplot, palette='mako')
            ax_countplot.set_title('Countplot')
            st.pyplot(fig_countplot)

            fig_boxplot, ax_boxplot = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=data, ax=ax_boxplot, palette='mako')
            ax_boxplot.set_title('Boxplot')
            st.pyplot(fig_boxplot)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader('Null amounts')
                null_info = pd.DataFrame(data.isnull().sum(), columns = ['null'])
                st.dataframe(null_info )

            with col2:
                st.subheader('Imputation')
                numerical_strategy = st.radio('Imputation for numerical', ['mean','most_frequent'])
                categorical_strategy = st.radio('Imputation for categorical', ['most_frequent'])
                st.subheader('Feature engineering')
                numerical_cols = data.select_dtypes(exclude = 'object').columns
                categorical_cols = data.select_dtypes(include = 'object').columns
                to_undersample = st.checkbox('UnderSampling(not implemented)')
                to_clean_outlier = st.checkbox('Clean outlier(not implemented)')
                if st.button('Run Data Preprocessing'):
                    preprocessor = ColumnTransformer([
                        ('cat', SimpleImputer(strategy= categorical_strategy), categorical_cols),
                        ('num', SimpleImputer(strategy= numerical_strategy), numerical_cols)
                    ],
                    remainder='passthrough')
                    data = pd.DataFrame(preprocessor.fit_transform(data), columns = categorical_cols.append(numerical_cols))

if page_choice == 'Modeling':
    st.dataframe(data)