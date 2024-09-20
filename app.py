import streamlit as st
from multiapp import MultiApp
from demandforecast import forecast_app
from eda import eda_app
from demandforecastweekly import demand_app 

st.set_page_config(layout="wide", page_title="Demand Forecast and Data Analysis", page_icon="logo.png")

app = MultiApp()

# Add all your application here
app.add_app("Forecast", forecast_app)
app.add_app("EDA", eda_app)
app.add_app("Weekly Forecast", demand_app)

# The main app
app.run()
   