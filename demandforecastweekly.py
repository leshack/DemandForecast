import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from prophet import Prophet
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import display, clear_output
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import re

sns.set_style("whitegrid")

def demand_app():
    st.title('DEMAND FORECASTING PIPELINE TOOL')
    st.markdown("""
        **Objective of Demand Forecasting:**

        - **Advise Production**: Provide recommendations on production quantities to align manufacturing output with forecasted demand.

        - **Optimize Inventory**: Ensure optimal inventory levels to meet customer demand while minimizing excess stock and reducing holding costs.

        - **Enhance Planning**: Support long-term strategic planning by predicting future demand trends, helping businesses make informed decisions about resources and investments.

        - **Improve Supply Chain Efficiency**: Align procurement, production schedules, and logistics with anticipated demand, reducing lead times and improving overall supply chain performance.

        - **Minimize Stockouts and Overstocks**: Balance supply and demand effectively to avoid shortages that can lead to lost sales and excess inventory that can increase carrying costs.

        - **Increase Customer Satisfaction**: Ensure that products are available when customers need them, enhancing service levels and maintaining customer loyalty.

        - **Support Financial Forecasting**: Provide data-driven insights for financial planning and budgeting, helping to forecast revenue and manage cash flow more effectively.
        """)
    
    st.markdown("""
        **Forecasting Models Used:**

        To achieve our forecasting objectives, we are leveraging four powerful models:

        - **Prophet**: A robust model designed to handle various patterns in time series data, including seasonality and trends, providing accurate and reliable forecasts.

        - **ARIMA**: A widely used method for time series forecasting that combines autoregressive and moving average components to model and predict future values based on historical data.

        - **Pre-trained ARIMA Model**: This model has been pre-trained and fitted with our specific data, ensuring it captures the unique patterns and trends relevant to our business needs.

        - **SARIMA**: An extension of ARIMA that includes seasonal components, allowing us to account for seasonal variations and provide forecasts that reflect periodic fluctuations in demand.

        By utilizing these models, we aim to create a comprehensive and accurate forecast that supports production planning, inventory management, and overall supply chain efficiency.
        """)
    # Function to parse dates
    def parse_dates(date_str):
        for fmt in ("%d/%m/%Y", "%m-%d-%Y", "%Y-%m-%d"):
            try:
                return pd.to_datetime(date_str, format=fmt, errors='coerce')
            except ValueError:
                continue
        return pd.NaT

    # Function to clean numeric columns
    def clean_column(df, column):
        # df[column] = df[column].str.replace(',', '', regex=True).astype(float)
        df[column] = df[column].astype(float)

    @st.cache_data
    # Function to load and preprocess data
    def load_and_preprocess_data(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
        df['Invoice Date'] = df['Invoice Date'].apply(parse_dates)
        df = df.dropna(subset=['Invoice Date'])  # Drop rows where Invoice Date is NaT
        clean_column(df, 'Value')
        clean_column(df, 'Quantity')
        df = df[df['Invoice Date'] >= '2021-01-01']  # Filter data from 2021 onward
        
        df['Year'] = df['Invoice Date'].dt.year
        # Create a 'Week' column based on the Invoice Date
        df['Week'] = df['Invoice Date'].dt.to_period('W').dt.start_time
       # Create a 'Week No.' column based on the ISO week number
        df['Week No.'] = df['Invoice Date'].dt.isocalendar().week
        
        return df

    # Function to aggregate data on a weekly basis
    @st.cache_data
    def aggregate_data(df):
        weekly_sales = df.groupby(['Year','Week', 'Week No.', 'Item Description', 'Colour Group', 'Classification'])[['Value', 'Quantity']].sum().reset_index()
        return weekly_sales

   
     
    # Load the pre-trained models and scaler
    # @st.cache_data
    # def load_model_data():
    #     with open('best_model.pkl', 'rb') as file:
    #         best_model = pickle.load(file)
    #     with open('scaler.pkl', 'rb') as file:
    #         scaler = pickle.load(file)
    #     data = pd.read_csv('Sales_invoices.csv')
    #     return best_model, scaler, data


    def forecast_arima(data, forecast_period):
        model = ARIMA(data, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_period)
        return forecast
    
    # SARIMA forecast function
    def forecast_sarima(data, forecast_period, order, seasonal_order):
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=forecast_period)
        return forecast
    
    def calculate_metrics(actual, predicted):
        mae = mean_absolute_error(actual, predicted)
        rmse = math.sqrt(mean_squared_error(actual, predicted))
        return mae, rmse

    def forecast_sales_and_quantities(weekly_sales, classification, item_description, colour_group, forecast_period, product_level, forecast_model):
        if product_level:
            group_data = weekly_sales[
                (weekly_sales['Item Description'] == item_description) &
                (weekly_sales['Classification'] == classification)
            ].groupby('Week').sum().reset_index()
        else:
            group_data = weekly_sales[
                (weekly_sales['Item Description'] == item_description) &
                (weekly_sales['Colour Group'] == colour_group) &
                (weekly_sales['Classification'] == classification)
            ]
        
        if group_data.dropna().shape[0] < 2:
            st.write("Not enough data for forecasting.")
            return

        prophet_data_sales = group_data[['Week', 'Value']].rename(columns={'Week': 'ds', 'Value': 'y'})
        prophet_data_quantity = group_data[['Week', 'Quantity']].rename(columns={'Week': 'ds', 'Quantity': 'y'})

        if forecast_model == 'Prophet':
            model_sales = Prophet()
            model_sales.fit(prophet_data_sales)
            future_sales = model_sales.make_future_dataframe(periods=forecast_period, freq='W')
            forecast_sales = model_sales.predict(future_sales)

            model_quantity = Prophet()
            model_quantity.fit(prophet_data_quantity)
            future_quantity = model_quantity.make_future_dataframe(periods=forecast_period, freq='W')
            forecast_quantity = model_quantity.predict(future_quantity)

            forecast_sales['yhat_quantity'] = forecast_quantity['yhat']
            
            st.header('Prophet Model')
            st.markdown("""
                [Prophet](https://towardsdatascience.com/time-series-from-scratch-decomposing-time-series-data-7b7ad0c30fe7) is a powerful open-source tool developed by **Facebook** for [time series forecasting](https://towardsdatascience.com/time-series-from-scratch-decomposing-time-series-data-7b7ad0c30fe7). It is designed to handle 
                various patterns in data, such as **seasonality, holidays, and trends**, making it ideal for business forecasting. 
                Prophet works by decomposing the time series into trend, seasonality, and holiday components, allowing for robust
                predictions even when data is irregular or contains missing points. With its ability to capture daily, weekly, or
                yearly seasonality, Prophet is highly versatile and provides accurate, reliable forecasts for **business insights and 
                decision-making**.
                """)

    
            st.markdown('### Sales forecast Sku vs Colour Wise')
            st.markdown("""
                    In our **demand forecasting**, we focus on **SKUs** where **color** is an important attribute, reflecting our diverse range of
                    plastic products. Since these products are classified as finished goods, capturing accurate forecasts by color helps 
                    in optimizing **production and inventory**. 
                    Using Prophet, we forecast sales trends and seasonal patterns for each SKU, providing insight into when demand 
                    peaks for specific colors. This allows us to better manage production planning and ensure we meet market needs 
                    effectively.
                        """)
            st.markdown(f"""
                    The forecast graph,Sales Forecast for {item_description} {"(All Colors)" if product_level else f"({colour_group})"}, 
                    shows predicted sales trends across different color variations, giving us a clearer picture 
                    of demand behavior at both individual and aggregated levels.
                    """)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(group_data['Week'], group_data['Value'], label='Actual Sales', color='blue',marker="o")
            ax.plot(forecast_sales['ds'], forecast_sales['yhat'], label='Forecast Sales', color='red')
            ax.fill_between(forecast_sales['ds'], forecast_sales['yhat_lower'], forecast_sales['yhat_upper'], alpha=0.2)
            ax.set_title(f'Sales Forecast for {item_description} {"(All Colors)" if product_level else f"({colour_group})"}')
            ax.set_xlabel('Weeks')
            ax.set_ylabel('Sales (Ksh)')
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%W'))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.legend()
            st.pyplot(fig)

            st.markdown('### Quantity forecast Sku vs Colour Wise')
            st.markdown("""
                Accurate forecasting of quantities is critical in optimizing both **production and inventory management**. 
                By predicting the quantity of products needed, we can provide valuable insights to production teams 
                on what to manufacture, ensuring that we meet demand without **overproducing**. 

                This level of precision also helps the warehouse teams in managing **storage efficiently**, preventing 
                excess inventory and reducing costs. When we forecast by SKU and color, it allows for a detailed 
                plan, ensuring the right quantities are produced and stored at the right time, which directly 
                supports **lean manufacturing principles and minimizes waste**.
                """)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(group_data['Week'], group_data['Quantity'], label='Actual Quantity', color='blue',marker="o")
            ax.plot(forecast_quantity['ds'], forecast_quantity['yhat'], label='Forecast Quantity', color='red')
            ax.fill_between(forecast_quantity['ds'], forecast_quantity['yhat_lower'], forecast_quantity['yhat_upper'], alpha=0.2)
            ax.set_title(f'Quantity Forecast for {item_description} {"(All Colors)" if product_level else f"({colour_group})"}')
            ax.set_xlabel('Weeks')
            ax.set_ylabel('Quantity')
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%W'))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.legend()
            st.pyplot(fig)

           # Determine the colour group value based on product_level
            display_colour_group = "(All Colors)" if product_level else f"({colour_group})"
           # Convert 'ds' (the date column) to the week number and year
            production_recommendation = forecast_quantity[['ds', 'yhat']].copy()
            
            # Extract Year and Week Number from the 'ds' column
            production_recommendation['Year'] = production_recommendation['ds'].dt.year
            production_recommendation['Week Number'] = production_recommendation['ds'].dt.isocalendar().week
            # Add item description and colour group to production recommendation
            production_recommendation['Item Description'] = item_description
            production_recommendation['Colour Group'] = display_colour_group
            # Rename the 'yhat' column to 'Recommended Production'
            production_recommendation = production_recommendation.rename(columns={'yhat': 'Recommended Production'})
            # Drop the 'ds' column (optional, if you only want the week number and not the date)
            production_recommendation = production_recommendation.drop(columns=['ds'])

            # Display the production recommendation in Streamlit
            #st.write('Recommended Production Quantities by Week:')
            #st.dataframe(production_recommendation)

            actual_production = group_data[['Week', 'Quantity']].rename(columns={'Week': 'ds', 'Quantity': 'y'})
            actual_production['Year'] = actual_production['ds'].dt.year
            actual_production['Week Number'] = actual_production['ds'].dt.isocalendar().week
            # Add item description and colour group to actual production
            actual_production['Item Description'] = item_description
            actual_production['Colour Group'] = display_colour_group
            # Rename the 'yhat' column to 'Recommended Production'
            actual_production = actual_production.rename(columns={'y': 'Actual Quantities Sold'})
            # Drop the 'ds' column (optional, if you only want the week number and not the date)
            actual_production = actual_production.drop(columns=['ds'])

            #st.dataframe(actual_production)

            # Merge production recommendation and actual production data based on Year and Week Number
            production_comparison = pd.merge(
            production_recommendation, 
            actual_production[['Year', 'Week Number', 'Actual Quantities Sold', 'Item Description', 'Colour Group']], 
            on=['Year', 'Week Number', 'Item Description', 'Colour Group'], 
            how='left'
               )

            # Display the combined comparison in Streamlit
            #st.write('Production Comparison (Recommended vs Actual):')
            st.markdown('### Production Comparison (Recommended vs Actual)')
            st.markdown("""
                    We generate a downloadable data frame that can be directly fed into the production schedule. 
                    This data frame includes the item description, color group, and the recommended production quantities 
                    for each week, which are crucial for advising the production team. The forecasted quantities are based 
                    on Prophet's predictions, giving us a clear view of expected demand for each SKU and color variation.

                    In addition, the actual quantities sold for previous months on weekly are included, allowing for a 
                    direct comparison between what was sold and what is recommended for the coming weeks. This 
                    comparison ensures that production stays aligned with demand and prevents overproduction or 
                    underproduction, making the process more efficient.

                    The production team can use this data to plan manufacturing output, while the warehouse team can 
                    optimize storage according to forecasted needs. This approach ensures smoother operations 
                    and better resource allocation across the entire supply chain.
                    """)

            st.dataframe(production_comparison)
            
            st.markdown('### Model Evaluation and Performance')
            st.markdown("""
                    In our forecasting model, we evaluate performance using two key metrics: **MAE (Mean Absolute Error)** 
                    and **RMSE (Root Mean Squared Error)**, both for sales and quantities.

                    - **Sales MAE**: This measures the average magnitude of the errors in our sales forecasts, 
                    without considering their direction. It tells us, on average, how much the forecasted sales 
                    values deviate from the actual sales values. A lower Sales MAE indicates that our model 
                    is making more accurate predictions of total revenue.

                    - **Quantity MAE**: Similar to Sales MAE, this metric focuses on the accuracy of quantity 
                    predictions. It reflects how closely our forecasted quantities match the actual number 
                    of units sold. This is critical for ensuring accurate production planning.

                    - **Sales RMSE**: RMSE gives more weight to larger errors by squaring the differences between 
                    forecasted and actual sales values. This metric is useful when we want to penalize 
                    large deviations, providing a sense of how well the model performs when there are significant 
                    errors in sales forecasting.

                    - **Quantity RMSE**: Like Sales RMSE, this focuses on the forecasted quantities but penalizes 
                    larger errors more heavily. It's essential for understanding how far off our forecasts are 
                    when there's a significant mismatch between predicted and actual quantities, helping us 
                    refine production schedules more effectively.

                    Both MAE and RMSE are used together to provide a balanced view of model performance, with MAE 
                    giving an overall sense of average error, while RMSE highlights how well the model handles 
                    larger errors.
                    """)


            # Performance metrics
            mae_sales, rmse_sales = calculate_metrics(group_data['Value'], forecast_sales['yhat'][:len(group_data)])
            mae_quantity, rmse_quantity = calculate_metrics(group_data['Quantity'], forecast_quantity['yhat'][:len(group_data)])
            
            st.markdown(f'Sales MAE: {mae_sales}, Sales RMSE: {rmse_sales}')
            st.markdown(f'Quantity MAE: {mae_quantity}, Quantity RMSE: {rmse_quantity}')

            st.toast('Prophet forecast successfully generated!')

        elif forecast_model == 'ARIMA':
            st.header('ARIMA Model')
            st.markdown("""
                The **[ARIMA](https://www.datacamp.com/tutorial/arima) (AutoRegressive Integrated Moving Average)** model is a popular and powerful forecasting 
                technique used for time series data. It works by combining three key elements: 

                 AR+I+MA= ARIMA       
                - **AutoRegressive (AR)**: This component uses the relationship between an observation and a number of 
                lagged (previous) observations to make predictions. It assumes that past values have an influence 
                on future outcomes.

                - **Integrated (I)**: This part accounts for the differencing of raw observations to make the time series 
                data stationary, meaning it removes trends and makes the data easier to predict.

                - **Moving Average (MA)**: The MA component captures the dependency between an observation and a residual 
                error from a moving average model applied to lagged observations.

                ARIMA is particularly useful when the data shows clear **patterns or seasonality**. Unlike other models, 
                ARIMA does not rely on external variables; it focuses purely on the **internal structure of the time 
                series data**. It can handle both short-term and long-term forecasting, making it valuable for predicting 
                demand and sales trends.

                ARIMA is often used alongside models like Prophet to offer additional insights and compare forecast 
                accuracy across multiple approaches.
                """)

    
            st.markdown('### Sales forecast Sku vs Colour Wise')
            st.markdown("""
                    In our **demand forecasting**, we focus on **SKUs** where **color** is a critical attribute, reflecting our wide range of 
                    plastic products. Since these products are classified as finished goods, forecasting accurately by color helps in optimizing 
                    **production and inventory management**.

                    Using ARIMA, we model the internal structure of sales data for each SKU, allowing us to identify trends and make precise 
                    predictions based on past patterns. This approach helps us forecast demand for specific colors, enabling us to better align 
                    our production schedules with market needs, while avoiding overproduction or stockouts.
                        """)
            st.markdown(f"""
                    The forecast graph,Sales Forecast for {item_description} {"(All Colors)" if product_level else f"({colour_group})"}, 
                    shows predicted sales trends across different color variations, giving us a clearer picture 
                    of demand behavior at both individual and aggregated levels.
                    """)
            forecast_sales_values = forecast_arima(group_data['Value'], forecast_period)
            forecast_quantity_values = forecast_arima(group_data['Quantity'], forecast_period)

            future_dates = pd.date_range(start=group_data['Week'].max(), periods=forecast_period+1, freq='W')[1:]

            forecast_sales = pd.DataFrame({
                'ds': future_dates,
                'yhat': forecast_sales_values
            })
            forecast_quantity = pd.DataFrame({
                'ds': future_dates,
                'yhat': forecast_quantity_values
            })  

            st.toast('ARIMA forecast successfully generated!')
            last_actual_sales = pd.DataFrame({
                'ds': [group_data['Week'].max()],
                'yhat': [group_data['Value'].iloc[-1]]
            })

            last_actual_quantity = pd.DataFrame({
                'ds': [group_data['Week'].max()],
                'yhat': [group_data['Quantity'].iloc[-1]]
            })

            forecast_sales_continuous = pd.concat([last_actual_sales, forecast_sales], ignore_index=True)
            forecast_quantity_continuous = pd.concat([last_actual_quantity, forecast_quantity], ignore_index=True)

            # Plot the updated continuous sales data
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot actual sales
            ax.plot(group_data['Week'], group_data['Value'], label='Actual Sales', color='blue', marker="o")

            # Plot forecast sales (continuous)
            ax.plot(forecast_sales_continuous['ds'], forecast_sales_continuous['yhat'], label='Forecast Sales(ARIMA)', color='red', marker="o")

            ax.set_title(f'Sales Forecast for {item_description} {"(All Colors)" if product_level else f"({colour_group})"}')
            ax.set_xlabel('Weeks')
            ax.set_ylabel('Sales (Ksh)')
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%W'))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.legend()
            st.pyplot(fig)

            st.markdown('### Quantity forecast Sku vs Colour Wise')
            st.markdown("""
                Accurate forecasting of quantities is critical in optimizing both **production and inventory management**. 
                By predicting the quantity of products needed, we can provide valuable insights to production teams 
                on what to manufacture, ensuring that we meet demand without **overproducing**. 

                This level of precision also helps the warehouse teams in managing **storage efficiently**, preventing 
                excess inventory and reducing costs. When we forecast by SKU and color, it allows for a detailed 
                plan, ensuring the right quantities are produced and stored at the right time, which directly 
                supports **lean manufacturing principles and minimizes waste**.
                """)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(group_data['Week'], group_data['Quantity'], label='Actual Quantity', color='blue',marker="o")
            ax.plot(forecast_quantity_continuous['ds'], forecast_quantity_continuous['yhat'], label='Forecast Quantity(ARIMA)', color='red', marker="o")
            ax.set_title(f'Quantity Forecast for {item_description} {"(All Colors)" if product_level else f"({colour_group})"}')
            ax.set_xlabel('Weeks')
            ax.set_ylabel('Quantity')
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%W'))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.legend()
            st.pyplot(fig)

            # Determine the colour group value based on product_level
            display_colour_group = "(All Colors)" if product_level else f"({colour_group})"
           # Convert 'ds' (the date column) to the week number and year
            production_recommendation = forecast_quantity_continuous[['ds', 'yhat']].copy()
            
            # Extract Year and Week Number from the 'ds' column
            production_recommendation['Year'] = production_recommendation['ds'].dt.year
            production_recommendation['Week Number'] = production_recommendation['ds'].dt.isocalendar().week
            # Add item description and colour group to production recommendation
            production_recommendation['Item Description'] = item_description
            production_recommendation['Colour Group'] = display_colour_group
            # Rename the 'yhat' column to 'Recommended Production'
            production_recommendation = production_recommendation.rename(columns={'yhat': 'Recommended Production'})
            # Drop the 'ds' column (optional, if you only want the week number and not the date)
            production_recommendation = production_recommendation.drop(columns=['ds'])

            # Display the production recommendation in Streamlit
            #st.write('Recommended Production Quantities by Week:')
            #st.dataframe(production_recommendation)

            actual_production = group_data[['Week', 'Quantity']].rename(columns={'Week': 'ds', 'Quantity': 'y'})
            actual_production['Year'] = actual_production['ds'].dt.year
            actual_production['Week Number'] = actual_production['ds'].dt.isocalendar().week
            # Add item description and colour group to actual production
            actual_production['Item Description'] = item_description
            actual_production['Colour Group'] = display_colour_group
            # Rename the 'yhat' column to 'Recommended Production'
            actual_production = actual_production.rename(columns={'y': 'Actual Quantities Sold'})
            # Drop the 'ds' column (optional, if you only want the week number and not the date)
            actual_production = actual_production.drop(columns=['ds'])

            #st.dataframe(actual_production)

            # Merge production recommendation and actual production data based on Year and Week Number
            production_comparison = pd.merge(
            production_recommendation, 
            actual_production[['Year', 'Week Number', 'Actual Quantities Sold', 'Item Description', 'Colour Group']], 
            on=['Year', 'Week Number', 'Item Description', 'Colour Group'], 
            how='left'
               )

            # Display the combined comparison in Streamlit
            #st.write('Production Comparison (Recommended vs Actual):')

            st.markdown('### Production Comparison (Recommended vs Actual)')
            st.markdown("""
                We generate a downloadable data frame that can be directly integrated into the production schedule. 
                This data frame contains the item description, color group, and the recommended production quantities 
                for each week, based on ARIMA’s predictions. These forecasts are essential for providing the production 
                team with clear guidance on expected demand for each SKU and color variation.

                In addition, the actual quantities sold from previous months on a weekly basis are included, allowing for 
                a direct comparison between historical sales and the recommended production for upcoming weeks. This 
                comparison ensures that production remains closely aligned with demand, preventing overproduction or 
                shortages and improving overall efficiency.

                The production team can use this data to adjust manufacturing output, while the warehouse team can 
                optimize storage space according to forecasted quantities. This approach helps streamline operations, 
                ensuring that resources are allocated effectively across the supply chain.
                """)
            st.dataframe(production_comparison)

            min_length = min(len(group_data['Value']), len(forecast_sales['yhat']))
            min_lengthQ = min(len(group_data['Quantity']), len(forecast_quantity['yhat']))

            # Adjust both arrays to the same length
            actual_sales = group_data['Value'][:min_length]
            predicted_sales = forecast_sales['yhat'][:min_length]
            actual_Q = group_data['Value'][:min_lengthQ]
            predicted_Q = forecast_quantity['yhat'][:min_lengthQ]

            st.markdown('### Model Evaluation and Performance')
            st.markdown("""
                    In our forecasting model, we evaluate performance using two key metrics: **MAE (Mean Absolute Error)** 
                    and **RMSE (Root Mean Squared Error)**, both for sales and quantities.

                    - **Sales MAE**: This measures the average magnitude of the errors in our sales forecasts, 
                    without considering their direction. It tells us, on average, how much the forecasted sales 
                    values deviate from the actual sales values. A lower Sales MAE indicates that our model 
                    is making more accurate predictions of total revenue.

                    - **Quantity MAE**: Similar to Sales MAE, this metric focuses on the accuracy of quantity 
                    predictions. It reflects how closely our forecasted quantities match the actual number 
                    of units sold. This is critical for ensuring accurate production planning.

                    - **Sales RMSE**: RMSE gives more weight to larger errors by squaring the differences between 
                    forecasted and actual sales values. This metric is useful when we want to penalize 
                    large deviations, providing a sense of how well the model performs when there are significant 
                    errors in sales forecasting.

                    - **Quantity RMSE**: Like Sales RMSE, this focuses on the forecasted quantities but penalizes 
                    larger errors more heavily. It's essential for understanding how far off our forecasts are 
                    when there's a significant mismatch between predicted and actual quantities, helping us 
                    refine production schedules more effectively.

                    Both MAE and RMSE are used together to provide a balanced view of model performance, with MAE 
                    giving an overall sense of average error, while RMSE highlights how well the model handles 
                    larger errors.
                    """)

            # Calculate metrics
            mae_sales, rmse_sales = calculate_metrics(actual_sales, predicted_sales)
            mae_quantity, rmse_quantity = calculate_metrics(actual_Q, predicted_Q)
            st.markdown(f'Sales MAE: {mae_sales}, Sales RMSE: {rmse_sales}')
            st.markdown(f'Quantity MAE: {mae_quantity}, Quantity RMSE: {rmse_quantity}')

        elif forecast_model == 'SARIMAX':
            st.header('SARIMAX Model')
            st.markdown("""
                The **[SARIMA](https://www.datacamp.com/tutorial/arima) (Seasonal AutoRegressive Integrated Moving Average)** model is an extension of ARIMA designed to handle seasonal patterns in time series data. It includes all the core components of ARIMA but adds a seasonal aspect to better capture recurring patterns over specific time intervals.

                    SAR + I + MA  = SARIMA

                - **Seasonal AutoRegressive (SAR)**: This component adds a seasonal autoregressive term that considers past values from the same season or period in previous cycles (e.g., weekly, monthly).
                                
                - **Seasonal Differencing (S)**: This part helps account for seasonality by differencing the data at the seasonal lag to make it stationary over longer-term periods.

                - **Seasonal Moving Average (SMA)**: Captures the relationship between an observation and the residual errors from a moving average model, applied to past observations in the same season or period.

                SARIMA is particularly useful when data exhibits **strong seasonal cycles** (such as monthly or yearly patterns) and can model both seasonality and trend simultaneously. This makes it ideal for forecasting sales or demand in businesses with clear seasonal demand spikes.

                SARIMA’s ability to model seasonality allows it to complement other models like ARIMA or Prophet, providing more **accurate and refined forecasts** when seasonality plays a significant role in the data.
                """)
            
            st.markdown('### Sales forecast SKU vs Colour Wise')
            st.markdown("""
                In our **demand forecasting**, we focus on **SKUs** where **color** is a key attribute, reflecting our diverse range of 
                plastic products. Since these products are classified as finished goods, accurately forecasting demand by color helps in 
                optimizing **production and inventory management**.

                Using SARIMA, we capture both the seasonal and non-seasonal components of sales data for each SKU. This model allows 
                us to account for recurring patterns and trends, giving us a more comprehensive view of demand cycles. By identifying 
                seasonal peaks and variations, we can better anticipate demand for specific colors, ensuring that our production schedules 
                align with market needs while minimizing both overproduction and stockouts.
            """)
            st.markdown(f"""
                The forecast graph, Sales Forecast for {item_description} {"(All Colors)" if product_level else f"({colour_group})"}, 
                highlights predicted sales trends across different color variations. It provides valuable insights into how seasonal and 
                non-seasonal demand patterns affect our products, helping us make more informed decisions about production and inventory levels.
            """)


            order = (1, 1, 1)  # Define SARIMA order
            seasonal_order = (1, 1, 1, 12)  # Define seasonal SARIMA order, e.g., yearly seasonality
            forecast_sales_values = forecast_sarima(group_data['Value'], forecast_period, order, seasonal_order)
            forecast_quantity_values = forecast_sarima(group_data['Quantity'], forecast_period, order, seasonal_order)

            future_dates = pd.date_range(start=group_data['Week'].max(), periods=forecast_period+1, freq='W')[1:]

            forecast_sales = pd.DataFrame({
                'ds': future_dates,
                'yhat': forecast_sales_values
            })
            forecast_quantity = pd.DataFrame({
                'ds': future_dates,
                'yhat': forecast_quantity_values
            })

            st.toast('SARIMA forecast successfully generated!')
            
            last_actual_sales = pd.DataFrame({
                'ds': [group_data['Week'].max()],
                'yhat': [group_data['Value'].iloc[-1]]
            })
            last_actual_quantity = pd.DataFrame({
                'ds': [group_data['Week'].max()],
                'yhat': [group_data['Quantity'].iloc[-1]]
            })

            forecast_sales_continuous = pd.concat([last_actual_sales, forecast_sales], ignore_index=True)
            forecast_quantity_continuous = pd.concat([last_actual_quantity, forecast_quantity], ignore_index=True)

            # Plotting sales forecast
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(group_data['Week'], group_data['Value'], label='Actual Sales', color='blue', marker="o")
            ax.plot(forecast_sales_continuous['ds'], forecast_sales_continuous['yhat'], label='Forecast Sales (SARIMA)', color='red', marker="o")
            ax.set_title(f'Sales Forecast for {item_description} {"(All Colors)" if product_level else f"({colour_group})"}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Sales (Ksh)')
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%W'))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.legend()
            st.pyplot(fig)

            st.markdown('### Quantity forecast SKU vs Colour Wise')
            st.markdown("""
                Accurate forecasting of quantities is crucial for optimizing both **production and inventory management**. 
                By predicting the quantity of products needed, we can ensure that our production aligns with market demand, 
                reducing the risks of **overproduction** or stock shortages.

                Using the SARIMA model, we account for both **seasonal** and **non-seasonal** factors in quantity forecasts. 
                This allows us to anticipate cyclical changes in demand for different colors and SKUs, ensuring that production 
                schedules and inventory levels are adjusted accordingly. Seasonal variations are particularly important when forecasting 
                quantities, as they help us plan for peaks and troughs in demand.

                This detailed forecasting approach helps the warehouse team manage **storage efficiently**, preventing excess inventory 
                and reducing holding costs. By aligning production with seasonal demand patterns, SARIMA allows us to apply **lean 
                manufacturing principles**, minimizing waste and ensuring that the right quantities are produced and stored at the 
                right time.
            """)

            # Plotting quantity forecast
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(group_data['Week'], group_data['Quantity'], label='Actual Quantity', color='blue', marker="o")
            ax.plot(forecast_quantity_continuous['ds'], forecast_quantity_continuous['yhat'], label='Forecast Quantity (SARIMA)', color='red', marker="o")
            ax.set_title(f'Quantity Forecast for {item_description} {"(All Colors)" if product_level else f"({colour_group})"}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Quantity')
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%W'))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.legend()
            st.pyplot(fig)

            # Determine the colour group value based on product_level
            display_colour_group = "(All Colors)" if product_level else f"({colour_group})"
            # Convert 'ds' (the date column) to the week number and year
            production_recommendation = forecast_quantity_continuous[['ds', 'yhat']].copy()
            
            # Extract Year and Week Number from the 'ds' column
            production_recommendation['Year'] = production_recommendation['ds'].dt.year
            production_recommendation['Week Number'] = production_recommendation['ds'].dt.isocalendar().week
            # Add item description and colour group to production recommendation
            production_recommendation['Item Description'] = item_description
            production_recommendation['Colour Group'] = display_colour_group
            # Rename the 'yhat' column to 'Recommended Production'
            production_recommendation = production_recommendation.rename(columns={'yhat': 'Recommended Production'})
            # Drop the 'ds' column (optional, if you only want the week number and not the date)
            production_recommendation = production_recommendation.drop(columns=['ds'])

            # Display the production recommendation in Streamlit
            #st.write('Recommended Production Quantities by Week:')
            #st.dataframe(production_recommendation)

            actual_production = group_data[['Week', 'Quantity']].rename(columns={'Week': 'ds', 'Quantity': 'y'})
            actual_production['Year'] = actual_production['ds'].dt.year
            actual_production['Week Number'] = actual_production['ds'].dt.isocalendar().week
            # Add item description and colour group to actual production
            actual_production['Item Description'] = item_description
            actual_production['Colour Group'] = display_colour_group
            # Rename the 'yhat' column to 'Recommended Production'
            actual_production = actual_production.rename(columns={'y': 'Actual Quantities Sold'})
            # Drop the 'ds' column (optional, if you only want the week number and not the date)
            actual_production = actual_production.drop(columns=['ds'])

            #st.dataframe(actual_production)

            # Merge production recommendation and actual production data based on Year and Week Number
            production_comparison = pd.merge(
            production_recommendation, 
            actual_production[['Year', 'Week Number', 'Actual Quantities Sold', 'Item Description', 'Colour Group']], 
            on=['Year', 'Week Number', 'Item Description', 'Colour Group'], 
            how='left'
               )
            
            st.markdown('### Production Comparison (Recommended vs Actual)')
            st.markdown("""
                We generate a downloadable data frame that can be directly integrated into the production schedule. 
                This data frame includes the item description, color group, and the recommended production quantities 
                for each week, based on SARIMA's predictions. These forecasts are essential for providing the production 
                team with precise guidance on expected demand, accounting for both seasonal and non-seasonal variations in 
                demand for each SKU and color variation.

                Additionally, the actual quantities sold from previous months on a weekly basis are included, allowing for 
                a direct comparison between historical sales and the recommended production for upcoming weeks. This 
                comparison helps to ensure that production remains aligned with demand patterns, reducing the risk of overproduction 
                or stock shortages while improving overall efficiency.

                The production team can use this data to adjust manufacturing output accordingly, while the warehouse team 
                can optimize storage and logistics planning based on forecasted quantities. This integrated approach improves 
                operations and ensures that resources are allocated effectively throughout the supply chain.
            """)
            st.dataframe(production_comparison)

            min_length = min(len(group_data['Value']), len(forecast_sales['yhat']))
            min_lengthQ = min(len(group_data['Quantity']), len(forecast_quantity['yhat']))

            # Adjust both arrays to the same length
            actual_sales = group_data['Value'][:min_length]
            predicted_sales = forecast_sales['yhat'][:min_length]
            actual_Q = group_data['Quantity'][:min_lengthQ]
            predicted_Q = forecast_quantity['yhat'][:min_lengthQ]

            st.markdown('### Model Evaluation and Performance')
            st.markdown("""
                    In our forecasting model, we evaluate performance using two key metrics: **MAE (Mean Absolute Error)** 
                    and **RMSE (Root Mean Squared Error)**, both for sales and quantities.

                    - **Sales MAE**: This measures the average magnitude of the errors in our sales forecasts, 
                    without considering their direction. It tells us, on average, how much the forecasted sales 
                    values deviate from the actual sales values. A lower Sales MAE indicates that our model 
                    is making more accurate predictions of total revenue.

                    - **Quantity MAE**: Similar to Sales MAE, this metric focuses on the accuracy of quantity 
                    predictions. It reflects how closely our forecasted quantities match the actual number 
                    of units sold. This is critical for ensuring accurate production planning.

                    - **Sales RMSE**: RMSE gives more weight to larger errors by squaring the differences between 
                    forecasted and actual sales values. This metric is useful when we want to penalize 
                    large deviations, providing a sense of how well the model performs when there are significant 
                    errors in sales forecasting.

                    - **Quantity RMSE**: Like Sales RMSE, this focuses on the forecasted quantities but penalizes 
                    larger errors more heavily. It's essential for understanding how far off our forecasts are 
                    when there's a significant mismatch between predicted and actual quantities, helping us 
                    refine production schedules more effectively.

                    Both MAE and RMSE are used together to provide a balanced view of model performance, with MAE 
                    giving an overall sense of average error, while RMSE highlights how well the model handles 
                    larger errors.
                    """)

            # Calculate metrics
            mae_sales, rmse_sales = calculate_metrics(actual_sales, predicted_sales)
            mae_quantity, rmse_quantity = calculate_metrics(actual_Q, predicted_Q)
            st.markdown(f'Sales MAE: {mae_sales}, Sales RMSE: {rmse_sales}')
            st.markdown(f'Quantity MAE: {mae_quantity}, Quantity RMSE: {rmse_quantity}')

            
        # elif forecast_model == 'Pre-trained Model':
                
        # # Prepare features for forecasting
        #     group_data['Week No.'] = group_data['Week'].dt.month
        #     group_data['Year'] = group_data['Week'].dt.year

        #     X = group_data[['Week No.', 'Year', 'Value', 'Quantity']]
        #     X_scaled = scaler.transform(X)

        #     # Predict using the pre-trained model
        #     forecast_sales_values = best_model.predict(X_scaled)
        #     forecast_quantity_values = best_model.predict(X_scaled)

        #     # Generate future dates
        #     future_dates = pd.date_range(start=group_data['Week'].max(), periods=forecast_period+1, freq='W')[1:]

        #     forecast_sales = pd.DataFrame({
        #         'ds': future_dates,
        #         'yhat': forecast_sales_values[-forecast_period:]
        #     })

        #     forecast_quantity = pd.DataFrame({
        #         'ds': future_dates,
        #         'yhat': forecast_quantity_values[-forecast_period:]
        #     })

        #     st.toast('Pre-trained Model forecast successfully generated!')

        #     fig, ax = plt.subplots(figsize=(10, 6))
        #     ax.plot(forecast_sales['ds'], forecast_sales['yhat'], label='Forecast')
        #     ax.set_title(f'Sales Forecast for {item_description} {"(All Colors)" if product_level else f"({colour_group})"}')
        #     ax.set_xlabel('Weeks')
        #     ax.set_ylabel('Sales (Ksh)')
        #     ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%W'))
        #     ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
        #     plt.xticks(rotation=45)
        #     plt.tight_layout()
        #     plt.legend()
        #     st.pyplot(fig)

        #     fig, ax = plt.subplots(figsize=(10, 6))
        #     ax.plot(forecast_quantity['ds'], forecast_quantity['yhat'], label='Forecast')
        #     ax.set_title(f'Quantity Forecast for {item_description} {"(All Colors)" if product_level else f"({colour_group})"}')
        #     ax.set_xlabel('Weeks')
        #     ax.set_ylabel('Quantity')
        #     ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%W'))
        #     ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
        #     plt.xticks(rotation=45)
        #     plt.tight_layout()
        #     plt.legend()
        #     st.pyplot(fig)
                  

    def forecast_overall_sales_and_quantities(weekly_sales, forecast_period, forecast_model):
        overall_data = weekly_sales.groupby('Week').sum().reset_index()
        
        if overall_data.dropna().shape[0] < 2:
            st.write("Not enough data for overall forecasting.")
            return
        
        prophet_data_sales = overall_data[['Week', 'Value']].rename(columns={'Week': 'ds', 'Value': 'y'})
        prophet_data_quantity = overall_data[['Week', 'Quantity']].rename(columns={'Week': 'ds', 'Quantity': 'y'})
        
        if forecast_model == 'Prophet':
            model_sales = Prophet()
            model_sales.fit(prophet_data_sales)
            future_sales = model_sales.make_future_dataframe(periods=forecast_period, freq='W')
            forecast_sales = model_sales.predict(future_sales)
            
            model_quantity = Prophet()
            model_quantity.fit(prophet_data_quantity)
            future_quantity = model_quantity.make_future_dataframe(periods=forecast_period, freq='W')
            forecast_quantity = model_quantity.predict(future_quantity)
            
            forecast_sales['yhat_quantity'] = forecast_quantity['yhat']

            st.markdown('### **Overall Weekly Sales Forecast**')
            st.markdown("""
                After applying our forecasting models, we analyze the projected overall sales for each week. This aggregated 
                sales data provides a comprehensive view of the expected revenue, allowing us to assess financial performance
                 and plan for future growth. By understanding weekly sales trends, we can make informed decisions about budgeting, 
                investment, and resource allocation. Accurate sales forecasts are crucial for aligning marketing strategies and 
                meeting revenue targets effectively.
                """)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(overall_data['Week'], overall_data['Value'], label='Actual Quantity', color='blue', marker="o")
            ax.plot(forecast_sales['ds'], forecast_sales['yhat'], label='Forecast Sales', color='red')
            ax.fill_between(forecast_sales['ds'], forecast_sales['yhat_lower'], forecast_sales['yhat_upper'], alpha=0.2)
            ax.set_title('Overall Sales Forecast')
            ax.set_xlabel('Weeks')
            ax.set_ylabel('Sales (Ksh)')
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%W'))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.legend()
            st.pyplot(fig)

            actual_recommendation = forecast_sales[['ds', 'yhat']].copy()
            
            # Extract Year and Week Number from the 'ds' column
            actual_recommendation['Year'] =  actual_recommendation['ds'].dt.year
            actual_recommendation['Week Number'] =  actual_recommendation['ds'].dt.isocalendar().week  
            # Rename the 'yhat' column to 'Recommended Production'
            actual_recommendation =  actual_recommendation.rename(columns={'yhat': 'Forecast Sales'})
            # Drop the 'ds' column (optional, if you only want the week number and not the date)
            actual_recommendation =  actual_recommendation.drop(columns=['ds'])

            actual_sale = overall_data[['Week', 'Value']].rename(columns={'Week': 'ds', 'Value': 'y'})
            actual_sale['Year'] = actual_sale['ds'].dt.year
            actual_sale['Week Number'] = actual_sale['ds'].dt.isocalendar().week
            
            # Rename the 'yhat' column to 'Recommended Production'
            actual_sale = actual_sale.rename(columns={'y': 'Actual realized Sales'})
            # Drop the 'ds' column (optional, if you only want the week number and not the date)
            actual_sale = actual_sale.drop(columns=['ds'])

            # Merge production recommendation and actual production data based on Year and Week Number
            production_comparison = pd.merge(
            actual_recommendation, 
            actual_sale[['Year', 'Week Number', 'Actual realized Sales']], 
            on=['Year', 'Week Number'], 
            how='left'
               )
            st.dataframe(production_comparison)
            
            st.markdown('### **Overall Weekly Quantities Forecast**')
            st.markdown("""
                In addition to sales, we also forecast the quantities of products needed for each week. This helps us determine 
                the exact number of units required to meet the anticipated demand. Accurate quantity forecasts are essential 
                for optimizing production schedules and inventory levels, preventing both stockouts and overstock situations. 
                By aligning production with forecasted quantities, we ensure that we can efficiently meet customer demand while
                minimizing waste and carrying costs.
                """)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(overall_data['Week'], overall_data['Quantity'], label='Actual Quantity', color='blue', marker="o")
            ax.plot(forecast_quantity['ds'], forecast_quantity['yhat'], label='Forecast Quantity', color='red')
            ax.fill_between(forecast_quantity['ds'], forecast_quantity['yhat_lower'], forecast_quantity['yhat_upper'], alpha=0.2)
            ax.set_title('Overall Quantity Forecast')
            ax.set_xlabel('Weeks')
            ax.set_ylabel('Quantity')
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%W'))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.legend()
            st.pyplot(fig)
            
            # Production recommendation
            # production_recommendation = forecast_sales[['ds', 'yhat']].rename(columns={'yhat': 'Forecast Sales'})
            # st.write('Recommended Production Quantities:')
            # st.dataframe(production_recommendation)
            
            # Performance metrics
            mae_sales, rmse_sales = calculate_metrics(overall_data['Value'], forecast_sales['yhat'][:len(overall_data)])
            mae_quantity, rmse_quantity = calculate_metrics(overall_data['Quantity'], forecast_quantity['yhat'][:len(overall_data)])
            
            st.markdown('### **Model Evaluation:**')
            st.markdown("""
            To ensure the effectiveness of our forecasting, we evaluate the performance of each model using key metrics such as 
            MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error). These metrics help us assess how well each model
            predicts both sales and quantities. By comparing the forecast accuracy of Prophet, ARIMA, the pre-trained ARIMA 
            model, and SARIMA, we can determine which model provides the most reliable forecasts. This evaluation allows us
            to refine our forecasting approach, improve accuracy, and make data-driven decisions to enhance overall business 
            performance.
            """)

            st.markdown(f'Sales MAE: {mae_sales}, Sales RMSE: {rmse_sales}')
            st.markdown(f'Quantity MAE: {mae_quantity}, Quantity RMSE: {rmse_quantity}')
        
        elif forecast_model == 'ARIMA':
            forecast_sales_values = forecast_arima(overall_data['Value'], forecast_period)
            forecast_quantity_values = forecast_arima(overall_data['Quantity'], forecast_period)
            
            future_dates = pd.date_range(start=overall_data['Week'].max(), periods=forecast_period+1, freq='W')[1:]
            
            forecast_sales = pd.DataFrame({
                'ds': future_dates,
                'yhat': forecast_sales_values
            })
            forecast_quantity = pd.DataFrame({
                'ds': future_dates,
                'yhat': forecast_quantity_values
            })

            last_actual_sales = pd.DataFrame({
                'ds': [overall_data['Week'].max()],
                'yhat': [overall_data['Value'].iloc[-1]]
            })

            last_actual_quantity = pd.DataFrame({
                'ds': [overall_data['Week'].max()],
                'yhat': [overall_data['Quantity'].iloc[-1]]
            })

            forecast_sales_continuous = pd.concat([last_actual_sales, forecast_sales], ignore_index=True)
            forecast_quantity_continuous = pd.concat([last_actual_quantity, forecast_quantity], ignore_index=True)

            st.markdown('### **Overall Weekly Sales Forecast**')
            st.markdown("""
                After applying our forecasting models, we analyze the projected overall sales for each week. This aggregated 
                sales data provides a comprehensive view of the expected revenue, allowing us to assess financial performance
                 and plan for future growth. By understanding weekly sales trends, we can make informed decisions about budgeting, 
                investment, and resource allocation. Accurate sales forecasts are crucial for aligning marketing strategies and 
                meeting revenue targets effectively.
                """)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(overall_data['Week'], overall_data['Value'], label='Actual Sales', color='blue', marker="o")
            ax.plot(forecast_sales_continuous['ds'], forecast_sales_continuous['yhat'], label='Forecast Sales(ARIMA)', color='red', marker="o")
            ax.set_title('Overall Sales Forecast')
            ax.set_xlabel('Weeks')
            ax.set_ylabel('Sales (Ksh)')
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%W'))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.legend()
            st.pyplot(fig)

            actual_recommendation = forecast_sales_continuous[['ds', 'yhat']].copy()
            
            # Extract Year and Week Number from the 'ds' column
            actual_recommendation['Year'] =  actual_recommendation['ds'].dt.year
            actual_recommendation['Week Number'] =  actual_recommendation['ds'].dt.isocalendar().week  
            # Rename the 'yhat' column to 'Recommended Production'
            actual_recommendation =  actual_recommendation.rename(columns={'yhat': 'Forecast Sales'})
            # Drop the 'ds' column (optional, if you only want the week number and not the date)
            actual_recommendation =  actual_recommendation.drop(columns=['ds'])

            actual_sale = overall_data[['Week', 'Value']].rename(columns={'Week': 'ds', 'Value': 'y'})
            actual_sale['Year'] = actual_sale['ds'].dt.year
            actual_sale['Week Number'] = actual_sale['ds'].dt.isocalendar().week
            
            # Rename the 'yhat' column to 'Recommended Production'
            actual_sale = actual_sale.rename(columns={'y': 'Actual realized Sales'})
            # Drop the 'ds' column (optional, if you only want the week number and not the date)
            actual_sale = actual_sale.drop(columns=['ds'])

            # Merge production recommendation and actual production data based on Year and Week Number
            production_comparison = pd.merge(
            actual_recommendation, 
            actual_sale[['Year', 'Week Number', 'Actual realized Sales']], 
            on=['Year', 'Week Number'], 
            how='left'
               )
            st.dataframe(production_comparison)

            st.markdown('### **Overall Weekly Quantities Forecast**')
            st.markdown("""
                In addition to sales, we also forecast the quantities of products needed for each week. This helps us determine 
                the exact number of units required to meet the anticipated demand. Accurate quantity forecasts are essential 
                for optimizing production schedules and inventory levels, preventing both stockouts and overstock situations. 
                By aligning production with forecasted quantities, we ensure that we can efficiently meet customer demand while
                minimizing waste and carrying costs.
                """)
            
            fig, ax = plt.subplots(figsize=(10, 6)) 
            ax.plot(overall_data['Week'], overall_data['Quantity'], label='Actual Quantity', color='blue', marker="o")
            ax.plot(forecast_quantity_continuous['ds'], forecast_quantity_continuous['yhat'], label='Forecast Quantity(ARIMA)', color='red', marker="o")
            ax.set_title('Overall Quantity Forecast')
            ax.set_xlabel('Weeks')
            ax.set_ylabel('Quantity')
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%W'))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.legend()
            st.pyplot(fig)

            # Production recommendation
            # production_recommendation = forecast_quantity[['ds', 'yhat']].rename(columns={'yhat': 'Recommended Production'})
            # st.write('Recommended Production Quantities:')
            # st.dataframe(production_recommendation)

            min_length = min(len(overall_data['Value']), len(forecast_sales['yhat']))
            min_lengthQ = min(len(overall_data['Quantity']), len(forecast_quantity['yhat']))

            # Adjust both arrays to the same length
            actual_sales = overall_data['Value'][:min_length]
            predicted_sales = forecast_sales['yhat'][:min_length]
            actual_Q = overall_data['Value'][:min_lengthQ]
            predicted_Q = forecast_quantity['yhat'][:min_lengthQ]

            st.markdown('### **Model Evaluation:**')
            st.markdown("""
            To ensure the effectiveness of our forecasting, we evaluate the performance of each model using key metrics such as 
            MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error). These metrics help us assess how well each model
            predicts both sales and quantities. By comparing the forecast accuracy of Prophet, ARIMA, the pre-trained ARIMA 
            model, and SARIMA, we can determine which model provides the most reliable forecasts. This evaluation allows us
            to refine our forecasting approach, improve accuracy, and make data-driven decisions to enhance overall business 
            performance.
            """)

            # Calculate metrics
            mae_sales, rmse_sales = calculate_metrics(actual_sales, predicted_sales)
            mae_quantity, rmse_quantity = calculate_metrics(actual_Q, predicted_Q)
            st.markdown(f'Sales MAE: {mae_sales}, Sales RMSE: {rmse_sales}')
            st.markdown(f'Quantity MAE: {mae_quantity}, Quantity RMSE: {rmse_quantity}')
        
        elif forecast_model == 'SARIMAX':

            order = (1, 1, 1)  # Define SARIMA order
            seasonal_order = (1, 1, 1, 12)  # Define seasonal SARIMA order, e.g., yearly seasonality
            forecast_sales_values = forecast_sarima(overall_data['Value'], forecast_period, order, seasonal_order)
            forecast_quantity_values = forecast_sarima(overall_data['Quantity'], forecast_period, order, seasonal_order)

            future_dates = pd.date_range(start=overall_data['Week'].max(), periods=forecast_period+1, freq='W')[1:]

            forecast_sales = pd.DataFrame({
                'ds': future_dates,
                'yhat': forecast_sales_values
            })
            forecast_quantity = pd.DataFrame({
                'ds': future_dates,
                'yhat': forecast_quantity_values
            })

            st.toast('SARIMA forecast successfully generated!')
            
            last_actual_sales = pd.DataFrame({
                'ds': [overall_data['Week'].max()],
                'yhat': [overall_data['Value'].iloc[-1]]
            })
            last_actual_quantity = pd.DataFrame({
                'ds': [overall_data['Week'].max()],
                'yhat': [overall_data['Quantity'].iloc[-1]]
            })

            forecast_sales_continuous = pd.concat([last_actual_sales, forecast_sales], ignore_index=True)
            forecast_quantity_continuous = pd.concat([last_actual_quantity, forecast_quantity], ignore_index=True)

            st.markdown('### **Overall Weekly Sales Forecast**')
            st.markdown("""
                After applying the SARIMA model, we analyze the projected overall sales for each week. This aggregated forecast 
                captures both seasonal and non-seasonal patterns, providing a detailed view of expected sales performance. By accounting 
                for cyclical demand fluctuations, we gain deeper insights into how different weeks are likely to perform in terms of revenue.

                This comprehensive outlook allows us to make informed decisions about budgeting, marketing strategies, and resource 
                allocation. Understanding these weekly sales trends helps us plan for future growth while ensuring that our production 
                schedules and inventory management align with market demand. Accurate sales forecasts are essential for achieving 
                revenue targets and maintaining financial stability.
            """)

            # Plotting sales forecast
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(overall_data['Week'], overall_data['Value'], label='Actual Sales', color='blue', marker="o")
            ax.plot(forecast_sales_continuous['ds'], forecast_sales_continuous['yhat'], label='Forecast Sales (SARIMA)', color='red', marker="o")
            ax.set_title('Overall Sales Forecast')
            ax.set_xlabel('Date')
            ax.set_ylabel('Sales (Ksh)')
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%W'))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.legend()
            st.pyplot(fig)

            actual_recommendation = forecast_sales_continuous[['ds', 'yhat']].copy()
            
            # Extract Year and month Number from the 'ds' column
            actual_recommendation['Year'] =  actual_recommendation['ds'].dt.year
            actual_recommendation['Week Number'] =  actual_recommendation['ds'].dt.isocalendar().week
            # Rename the 'yhat' column to 'Recommended Production'
            actual_recommendation =  actual_recommendation.rename(columns={'yhat': 'Forecast Sales'})
            # Drop the 'ds' column (optional, if you only want the month number and not the date)
            actual_recommendation =  actual_recommendation.drop(columns=['ds'])

            actual_sale = overall_data[['Week', 'Value']].rename(columns={'Week': 'ds', 'Value': 'y'})
            actual_sale['Year'] = actual_sale['ds'].dt.year
            actual_sale['Week Number'] = actual_sale['ds'].dt.isocalendar().week
            
            # Rename the 'yhat' column to 'Recommended Production'
            actual_sale = actual_sale.rename(columns={'y': 'Actual realized Sales'})
            # Drop the 'ds' column (optional, if you only want the month number and not the date)
            actual_sale = actual_sale.drop(columns=['ds'])

            # Merge production recommendation and actual production data based on Year and month Number
            production_comparison = pd.merge(
            actual_recommendation, 
            actual_sale[['Year', 'Week Number', 'Actual realized Sales']], 
            on=['Year', 'Week Number'], 
            how='left'
               )
            st.dataframe(production_comparison)

            st.markdown('### **Overall Weekly Quantities Forecast**')
            st.markdown("""
                In addition to sales, we use the SARIMA model to forecast the quantities of products required for each week. This forecast 
                accounts for both seasonal trends and fluctuations in demand, helping us determine the exact number of units needed 
                to meet anticipated market requirements.

                Accurate quantity forecasts are critical for optimizing production schedules and managing inventory levels. By predicting 
                weekly quantities, we can ensure that production aligns with demand, preventing both stockouts and overstock situations. 
                This allows us to meet customer needs efficiently while minimizing waste and reducing carrying costs, ultimately supporting 
                lean production and improving overall operational efficiency.
            """)

            # Plotting quantity forecast
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(overall_data['Week'], overall_data['Quantity'], label='Actual Quantity', color='blue', marker="o")
            ax.plot(forecast_quantity_continuous['ds'], forecast_quantity_continuous['yhat'], label='Forecast Quantity (SARIMA)', color='red', marker="o")
            ax.set_title('Overall Quantity Forecast')
            ax.set_xlabel('Date')
            ax.set_ylabel('Quantity')
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%W'))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.legend()
            st.pyplot(fig)

            st.markdown('### **Model Evaluation:**')
            st.markdown("""
            To ensure the effectiveness of our forecasting, we evaluate the performance of each model using key metrics such as 
            MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error). These metrics help us assess how well each model
            predicts both sales and quantities. By comparing the forecast accuracy of Prophet, ARIMA, the pre-trained ARIMA 
            model, and SARIMA, we can determine which model provides the most reliable forecasts. This evaluation allows us
            to refine our forecasting approach, improve accuracy, and make data-driven decisions to enhance overall business 
            performance.
            """)

            # Calculate metrics
            mae_sales, rmse_sales = calculate_metrics(actual_sales, predicted_sales)
            mae_quantity, rmse_quantity = calculate_metrics(actual_Q, predicted_Q)
            st.markdown(f'Sales MAE: {mae_sales}, Sales RMSE: {rmse_sales}')
            st.markdown(f'Quantity MAE: {mae_quantity}, Quantity RMSE: {rmse_quantity}')

        # elif forecast_model == 'Pre-trained Model':
            
        #         # Prepare features for forecasting
        #         overall_data['Week No.'] = overall_data['Week'].dt.month
        #         overall_data['Year'] = overall_data['Week'].dt.year
    
        #         X = overall_data[['Week No.', 'Year', 'Value', 'Quantity']]
        #         X_scaled = scaler.transform(X)
    
        #         # Predict using the pre-trained model
        #         forecast_sales_values = best_model.predict(X_scaled)
        #         forecast_quantity_values = best_model.predict(X_scaled)
    
        #         # Generate future dates
        #         future_dates = pd.date_range(start=overall_data['Week'].max(), periods=forecast_period+1, freq='W')[1:]
    
        #         forecast_sales = pd.DataFrame({
        #             'ds': future_dates,
        #             'yhat': forecast_sales_values[-forecast_period:]
        #         })
    
        #         forecast_quantity = pd.DataFrame({
        #             'ds': future_dates,
        #             'yhat': forecast_quantity_values[-forecast_period:]
        #         })
    
        #         st.toast('Pre-trained Model forecast successfully generated!')
    
        #         fig, ax = plt.subplots(figsize=(10, 6))
        #         ax.plot(forecast_sales['ds'], forecast_sales['yhat'], label='Forecast')
        #         ax.set_title('Overall Sales Forecast (Pre-trained Model)')  
        #         ax.set_xlabel('Weeks')
        #         ax.set_ylabel('Sales (Ksh)')
        #         ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%W'))
        #         ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
        #         plt.xticks(rotation=45)
        #         plt.tight_layout()
        #         plt.legend()
        #         st.pyplot(fig)
    
        #         fig, ax = plt.subplots(figsize=(10, 6))
        #         ax.plot(forecast_quantity['ds'], forecast_quantity['yhat'], label='Forecast')
        #         ax.set_title('Overall Quantity Forecast (Pre-trained Model)')
        #         ax.set_xlabel('Weeks')
        #         ax.set_ylabel('Quantity')
        #         ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%W'))
        #         ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
        #         plt.xticks(rotation=45)
        #         plt.tight_layout()
        #         plt.legend()
        #         st.pyplot(fig)

    # Load and preprocess data
    #csv_path = 'sales_invoices.csv'
    csv_path = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    if csv_path is not None:
        df = load_and_preprocess_data(csv_path)

        # Aggregate data by week
        weekly_sales = aggregate_data(df)
        st.header('Aggregation to Weekly')
        st.markdown("""
                To improve forecasting accuracy, we've aggregated the sales data on a weekly basis. By doing this, we smooth out daily 
                fluctuations and capture consistent buying patterns that are more relevant to planning and decision-making. Weekly
                aggregation also allows for better trend identification and seasonality analysis, which are crucial for generating 
                reliable demand forecasts. This approach ensures we make data-driven predictions that align with real-world sales cycles 
                and customer behaviors.
                """)

        st.write(weekly_sales)
        # best_model, scaler, data = load_model_data()

        # Sidebar Widgets
        st.sidebar.header('Forecast Options')
        # Date selection widgets
        
        
        classification_options = df['Classification'].unique().tolist()
        classification = st.sidebar.selectbox('Classification:', classification_options)

        item_description_options = df[df['Classification'] == classification]['Item Description'].unique().tolist()
        item_description = st.sidebar.selectbox('Item Description:', item_description_options)

        color_group_options = df[(df['Classification'] == classification) & 
                                (df['Item Description'] == item_description)]['Colour Group'].unique().tolist()
        colour_group = st.sidebar.selectbox('Colour Group:', color_group_options)

        forecast_period = st.sidebar.slider('Forecast Period (weeks):', min_value=1, max_value=24, value=12)

        product_level = st.sidebar.checkbox('Product Level Forecast')

        forecast_model = st.sidebar.selectbox('Forecast Model:', ['Prophet', 'ARIMA','SARIMAX','Pre-trained Model'])
    else:
        st.warning("Please upload a CSV file to proceed.")

    if st.sidebar.button('Generate Forecast'):
        forecast_sales_and_quantities(weekly_sales, classification, item_description, colour_group, forecast_period, product_level, forecast_model)
        forecast_overall_sales_and_quantities(weekly_sales, forecast_period, forecast_model)






     
