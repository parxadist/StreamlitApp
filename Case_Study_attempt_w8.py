import streamlit as st
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


def outlier_cleaner(col):
    q1_pos, q3_pos = np.percentile(col,[25,75])
    iqr = q3_pos - q1_pos
    lower_fence = q1_pos - 1.5*iqr
    higher_fence = q3_pos + 1.5*iqr
    cleaned_col = col[(col<=higher_fence) & (col>=lower_fence)]
    return cleaned_col
    

page_choice = st.sidebar.selectbox('Page Choice',['HomePage','EDA','Modeling'], )
selected_dataset = st.selectbox("Select dataset you want",['Loan Dataset', 'Water Dataset'])
data = None
preprocessed_data = None

if page_choice == 'HomePage':
    if os.path.exists('model.csv'):
        os.remove('model.csv')
    st.header('Attempt of creating prediction site')
    st.image("DataImg.png")
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

            null_info = pd.DataFrame(data.isnull().sum(), columns = ['null'])

            col1.subheader('Null amounts')
            col1.dataframe(null_info)

            
            col2.subheader('Imputation')
            numerical_strategy = col2.radio('Imputation for numerical', ['mean','most_frequent'])
            categorical_strategy = col2.radio('Imputation for categorical', ['most_frequent'])
            col2.subheader('Feature engineering')
            numerical_cols = data.select_dtypes(exclude = 'object').columns
            categorical_cols = data.select_dtypes(include = 'object').columns
            to_undersample = col2.checkbox('UnderSampling(not implemented)')
            to_clean_outlier = col2.checkbox('Clean outlier')
            data['Credit_History'].fillna(1, inplace = True)
            if col2.button('Run Data Preprocessing'):
                preprocessor = ColumnTransformer([
                    ('cat', SimpleImputer(strategy= categorical_strategy), categorical_cols),
                    ('num', SimpleImputer(strategy= numerical_strategy), numerical_cols)
                ],
                remainder='passthrough')
                preprocessed_data = pd.DataFrame(preprocessor.fit_transform(data), columns = categorical_cols.append(numerical_cols))
                preprocessed_data['LoanAmount'] = preprocessed_data['LoanAmount'].apply(round)
                preprocessed_data = pd.DataFrame(preprocessor.fit_transform(data), columns = categorical_cols.append(numerical_cols))

                if to_clean_outlier:
                    preprocessed_data[numerical_cols] = preprocessed_data[numerical_cols].apply(outlier_cleaner)
                    preprocessed_data.dropna(axis= 0, inplace = True)
                    fig_countplot, ax_countplot = plt.subplots(figsize=(8, 5))
                    sns.countplot(data=preprocessed_data, x='Loan_Status', ax=ax_countplot, palette='mako')
                    ax_countplot.set_title('Countplot')
                    st.pyplot(fig_countplot)
                if os.path.exists('model.csv'):
                    os.remove('model.csv')
                preprocessed_data.to_csv('model.csv')
                
                st.success('Model is preprocessed, proceed to the next page')


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
                to_clean_outlier = st.checkbox('Clean outlier')
                if st.button('Run Data Preprocessing'):
                    preprocessor = ColumnTransformer([
                        ('cat', SimpleImputer(strategy= categorical_strategy), categorical_cols),
                        ('num', SimpleImputer(strategy= numerical_strategy), numerical_cols)
                    ],
                    remainder='passthrough')
                    preprocessed_data = pd.DataFrame(preprocessor.fit_transform(data), columns = categorical_cols.append(numerical_cols))
                    if os.path.exists('model.csv'):
                        os.remove('model.csv')
                    preprocessed_data.to_csv('model.csv')
                    
if page_choice == 'Modeling':
    if(not os.path.exists('model.csv')):
       st.warning('You need to preprocess data in EDA first')
    else:
        data = pd.read_csv('model.csv').drop('Unnamed: 0', axis = 1)
        st.header('Preprocessed Data')
        st.dataframe(data)
        st.dataframe(data.isnull().sum())
        col3, col4 = st.columns(2)
        with col3:
            scaler_choice = st.radio('Scaler Choice',['Minmax','Standard','Robust'])
        with col4:
            encoding_choice = st.radio('Encoding Choice', ['OneHotEncoder','LabelEncoder'])
        st.header('Train test split:')
        train_size = float(st.slider(label = 'Train data size',min_value = 0.5, max_value = 0.9, value=0.8))
        train_model = st.selectbox('Model to train', ['XGBoost'])
        if st.button('Train and Run Model'):
            data = pd.read_csv('model.csv')
            data.drop('Unnamed: 0', axis = 1, inplace= True)
            # Step 2: Split the dataset into features (X) and target variable (y)
            X = data.iloc[:,:-1]  
            y = data.iloc[:,-1]

            # Step 3: Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            #st.text(str(X_train.shape) + ' ' + str(X_test.shape))
            #st.text(str(y_train.shape) + ' ' + str(y_test.shape))

            # Step 4: Identify categorical and numerical columns
            categorical_columns = X_train.select_dtypes(include=['object']).columns
            numerical_columns = X_train.select_dtypes(exclude=['object']).columns
            
            # Step 5: One-Hot Encode categorical data
            match encoding_choice:
                case 'OneHotEncoder':
                    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output = False, drop = 'first')
                case 'LabelEncoder':
                    encoder = LabelEncoder()

            X_train_categorical = pd.DataFrame(encoder.fit_transform(X_train[categorical_columns]))
            X_test_categorical = pd.DataFrame(encoder.transform(X_test[categorical_columns]))
            st.text(str(X_train_categorical.shape) + ' ' + str(X_test_categorical.shape))
            # Assign column names to the one-hot encoded features
            X_train_categorical.columns = encoder.get_feature_names_out(categorical_columns)
            X_test_categorical.columns = encoder.get_feature_names_out(categorical_columns)
            
            #st.dataframe(X_train_categorical)
            #st.dataframe(X_train[numerical_columns])
            #st.dataframe(X_test_categorical)
            #st.dataframe(X_test[numerical_columns])

            # Step 6: Concatenate one-hot encoded features with numerical features
            X_train_encoded = pd.concat([X_train_categorical, X_train[numerical_columns].reset_index()], axis=1)
            X_test_encoded = pd.concat([X_test_categorical, X_test[numerical_columns].reset_index()], axis=1)

            #st.dataframe(X_train_encoded)
            #st.dataframe(X_test_encoded)

            # Step 7: Scale numerical data
            match scaler_choice:
                case 'Minmax':
                    scaler = MinMaxScaler()
                case 'Standard':
                    scaler = StandardScaler()
                case 'Robust':
                    scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_encoded)
            X_test_scaled = scaler.transform(X_test_encoded)

            xgmodel = XGBClassifier().fit(X_train_scaled, y_train)
            y_pred = xgmodel.predict(X_test_scaled)

            st.text(classification_report(y_pred, y_test))