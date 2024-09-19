import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle


sns.set_style("whitegrid")

def eda_app():

    #st.header('Exploratory Data Analysis (EDA)')

    # Function to parse dates
    def parse_dates(date_str):
        for fmt in ("%d/%m/%Y", "%m-%d-%Y", "%Y-%m-%d"):
            try:
                return pd.to_datetime(date_str, format=fmt, errors='coerce')
            except ValueError:
                continue
        return pd.NaT

    # Function to clean columns
    def clean_column(df, column):
        df[column] = df[column].str.replace(',', '', regex=True).astype(float)

    # Function to remove outliers using IQR
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # Function to load and preprocess data
    def load_and_preprocess_data(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
        df['Invoice Date'] = df['Invoice Date'].apply(parse_dates)
        df = df.dropna(subset=['Invoice Date'])
        clean_column(df, 'Value')
        clean_column(df, 'Quantity')
        df = df[df['Invoice Date'] >= '2021-01-01']
        df['Month'] = df['Invoice Date'].dt.to_period('M').dt.to_timestamp()

        df = remove_outliers(df, 'Value')
        df = remove_outliers(df, 'Quantity')

        columns_to_drop = ['Account Manager', 'DISC %']
        df = df.drop(columns=columns_to_drop, errors='ignore')

        return df

    # Load data
    st.title("Sales Data Analysis & Modeling")
    csv_path = st.file_uploader("Upload your CSV file", type="csv")
    if csv_path is not None:
        df = load_and_preprocess_data(csv_path)
        st.write("Data Preview:", df.head())
            
        # Aggregating data
        st.subheader("Monthly Sales Aggregation")
        monthly_sales = df.groupby(['Month', 'Item Description', 'Colour Group'])[['Value', 'Quantity']].sum().reset_index()
        st.write("Monthly Sales Data", monthly_sales)

        # Descriptive statistics
        st.subheader("Descriptive Statistics")
        st.write(df.describe(include='all'))

        # Checking for missing data
        st.subheader("Missing Data")
        st.write(df.isnull().sum())
        st.write("Missing Data Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        st.pyplot()

        # Visualizations
        st.subheader("Exploratory Data Analysis")
        st.write("Histograms")
        df.hist(figsize=(40, 20), bins=20, color='blue')
        st.pyplot()

        st.write("Box Plots")
        numeric_columns = df.select_dtypes(include=['float', 'int']).columns
        for column in numeric_columns:
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=df[column])
            plt.title(f'Boxplot of {column}')
            st.pyplot()

        st.write("Correlation Heatmap")
        plt.figure(figsize=(15, 10))
        sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
        st.pyplot()

        # Feature Engineering
        df['Month No.'] = df['Invoice Date'].dt.month
        df['Year'] = df['Invoice Date'].dt.year

        # Selecting features
        features = ['Month No.', 'Year', 'Value', 'Quantity']
        target = 'Value'
        X = df[features]
        y = df[target]

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Model training and evaluation
        st.subheader("Model Training & Evaluation")
        rf = RandomForestRegressor()
        cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
        st.write(f'Random Forest Cross-Validation Scores: {cv_scores_rf}')

        # Hyperparameter Tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        st.write(f'Best Model: {best_model}')

        # Model evaluation
        def evaluate_model(model, X_train, X_test, y_train, y_test):
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            st.write("Train Performance")
            st.write(f"Mean Squared Error: {mean_squared_error(y_train, y_train_pred)}")
            st.write(f"Mean Absolute Error: {mean_absolute_error(y_train, y_train_pred)}")
            st.write(f"R^2 Score: {r2_score(y_train, y_train_pred)}")

            st.write("\nTest Performance")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_test_pred)}")
            st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_test_pred)}")
            st.write(f"R^2 Score: {r2_score(y_test, y_test_pred)}")

        evaluate_model(best_model, X_train, X_test, y_train, y_test)

        # Save the model and scaler
        with open('best_model.pkl', 'wb') as file:
            pickle.dump(best_model, file)
        with open('scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)

        st.success("Model and scaler saved successfully!")

    else:
        st.warning("Please upload a CSV file to proceed.")


